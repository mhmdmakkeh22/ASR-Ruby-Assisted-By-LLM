require "numo/narray"
require "fileutils"
require_relative "logsumexp"
require_relative "gmm"

module ASR
  module Math
    class HMMGMM
      LOG_ZERO = -1e10

      attr_reader :n_states, :gmms, :pi, :a

      def initialize(n_states, dim, n_components: 4)
        @n_states = n_states
        @dim = dim
        @gmms = Array.new(n_states) { GMM.new(n_components, dim) }
        @pi = Numo::DFloat.zeros(n_states).fill(1.0 / n_states)
        @a = Numo::DFloat.zeros(n_states, n_states)
        (0...n_states).each { |i| @a[i, true] = Numo::DFloat.zeros(n_states).fill(1.0 / n_states) }
      end

      # Train the HMM-GMM model.
      # sequences: Array of Numo::DFloat [T,D]
      # logger: nil | path string | IO-like object
      def train(sequences, n_iter: 5, em_iters_per_state: 3, logger: nil, checkpoint_dir: nil, dev_sequences: nil)
        log_io = nil
        if logger
          if logger.is_a?(String)
            begin
              log_io = File.open(logger, "a")
            rescue => _e
              log_io = nil
            end
          elsif logger.respond_to?(:write)
            log_io = logger
          end
        end

        log = ->(s) do
          puts s
          STDOUT.flush
          begin
            log_io.write(s + "\n") if log_io
            log_io.flush if log_io
          rescue => _e
          end
        end

        # Initial global fit
        all = concat_sequences(sequences)
        @gmms.each { |g| g.fit(all, n_iter: em_iters_per_state) }

        log.call("Initial per-state GMM summaries:")
        @gmms.each_with_index do |g, s|
          log.call("State #{s}:")
          log.call(g.summary)
        end

        prev_avg_loglik = nil

        n_iter.times do |it|
          iter_no = it + 1
          iter_label = "#{iter_no}/#{n_iter}"
          iter_start = Time.now
          log.call("HMM-GMM: iteration #{iter_label} starting at #{iter_start.strftime('%Y-%m-%d %H:%M:%S')} — EM iters per state=#{em_iters_per_state}")

          gamma_sum = Numo::DFloat.zeros(@n_states)
          xi_sum    = Numo::DFloat.zeros(@n_states, @n_states)
          state_frames = Array.new(@n_states) { [] }
          total_loglik = 0.0

          sequences.each do |x|
            log_b = emission_log_likelihoods(x)
            log_alpha, log_l = forward(log_b)
            log_beta = backward(log_b)
            gamma, xi = posteriors(log_alpha, log_beta, log_b, log_l)

            total_loglik += log_l

            gamma_sum += gamma.sum(0)
            xi_sum    += xi.sum(0)

            labels = gamma.max_index(1)
            # Group frames by most likely state using boolean masks (Numo operations)
            (0...@n_states).each do |s|
              mask = labels.eq(s)
              next if mask.where.empty?
              chunk = x[mask, true]
              # append each row (as Ruby array) so later Numo::DFloat.cast reshapes cleanly
              chunk.to_a.each { |row| state_frames[s] << row }
            end
          end

          # Update initial distribution.
          @pi = gamma_sum / gamma_sum.sum

          # Update transition probabilities.
          (0...@n_states).each do |i|
            row = xi_sum[i, true]
            next if row.sum <= 0
            row = row / row.sum
            row = row.clip(1e-8, 1.0)
            @a[i, true] = row / row.sum
          end

          avg_loglik = total_loglik / sequences.length
          if prev_avg_loglik
            delta = avg_loglik - prev_avg_loglik
            log.call("  avg log-likelihood: #{avg_loglik.round(4)} (Δ #{delta.round(6)})")
          else
            log.call("  avg log-likelihood: #{avg_loglik.round(4)}")
          end
          prev_avg_loglik = avg_loglik

          # Re-fit GMMs per state using assigned frames.
          @n_states.times do |s|
            next if state_frames[s].empty?
            # Ensure each row has correct dimensionality before casting
            rows = state_frames[s].map do |r|
              rr = r.respond_to?(:to_a) ? r.to_a : Array(r)
              rr = rr.flatten
              if rr.length != @dim
                rr = rr[0...@dim] + [0.0] * [0, (@dim - rr.length)].max
              end
              rr
            end
            x_s = Numo::DFloat.cast(rows)
            log.call("  Re-fitting GMM for state #{s} with #{x_s.shape[0]} frames")
            @gmms[s].fit(x_s, n_iter: em_iters_per_state)
            log.call("  Updated GMM for state #{s}:")
            log.call(@gmms[s].summary)
          end

          # optional dev set scoring
          if dev_sequences && !dev_sequences.empty?
            dev_score = score_sequences(dev_sequences)
            log.call("  dev avg log-likelihood: #{dev_score.round(4)}")
          end

          # checkpoint model after iteration
          if checkpoint_dir
            begin
              FileUtils.mkdir_p(checkpoint_dir)
              ckpt_path = File.join(checkpoint_dir, "hmm_gmm_model.iter#{iter_no}.marshal")
              File.open(ckpt_path, "wb") { |f| f.write(Marshal.dump(self)) }
              log.call("  checkpoint saved: #{ckpt_path}")
            rescue => _e
              log.call("  WARNING: failed to write checkpoint")
            end
          end

          iter_end = Time.now
          duration = iter_end - iter_start
          log.call("HMM-GMM: iteration #{iter_label} finished at #{iter_end.strftime('%Y-%m-%d %H:%M:%S')} (duration: #{duration.round(2)}s)")
        end

        self
      end

      # Viterbi decoding: returns most likely state sequence (Array of Integers).
      def viterbi(x)
        t_len = x.shape[0]
        log_b = emission_log_likelihoods(x)
        delta = Numo::DFloat.zeros(t_len, @n_states) + LOG_ZERO
        psi   = Numo::Int32.zeros(t_len, @n_states)

        @n_states.times do |i|
          delta[0, i] = ::Math.log(@pi[i]) + log_b[0, i]
          psi[0, i] = 0
        end

        (1...t_len).each do |t|
          @n_states.times do |j|
            scores = Numo::DFloat.zeros(@n_states) + LOG_ZERO
            @n_states.times do |i|
              next if @a[i, j] <= 0
              scores[i] = delta[t - 1, i] + ::Math.log(@a[i, j])
            end
            best = scores.max_index
            delta[t, j] = scores[best] + log_b[t, j]
            psi[t, j] = best
          end
        end

        path = Array.new(t_len)
        last = delta[t_len - 1, true].max_index
        path[-1] = last
        (t_len - 2).downto(0) do |t|
          path[t] = psi[t + 1, path[t + 1]]
        end
        path
      end

      private

      # Concatenate variable-length [T,D] sequences into one big [T_total,D] matrix.
      def concat_sequences(sequences)
        dim = sequences.first.shape[1]
        total = sequences.inject(0) { |acc, x| acc + x.shape[0] }
        all = Numo::DFloat.zeros(total, dim)
        offset = 0
        sequences.each do |x|
          t = x.shape[0]
          all[offset...(offset + t), true] = x
          offset += t
        end
        all
      end

      # Emission log-likelihoods: returns Numo::DFloat[T, @n_states]
      def emission_log_likelihoods(x)
        t_len = x.shape[0]
        log_b = Numo::DFloat.zeros(t_len, @n_states)
        @n_states.times do |s|
          log_b[true, s] = @gmms[s].logpdf(x)
        end
        log_b
      end

      # Forward algorithm in log-domain.
      # log_b: [T, S], log emission probabilities
      # Returns [log_alpha (T,S), log_likelihood]
      def forward(log_b)
        t_len = log_b.shape[0]
        log_alpha = Numo::DFloat.zeros(t_len, @n_states) + LOG_ZERO

        @n_states.times do |i|
          log_alpha[0, i] = ::Math.log(@pi[i]) + log_b[0, i]
        end

        (1...t_len).each do |t|
          @n_states.times do |j|
            scores = Numo::DFloat.zeros(@n_states) + LOG_ZERO
            @n_states.times do |i|
              next if @a[i, j] <= 0
              scores[i] = log_alpha[t - 1, i] + ::Math.log(@a[i, j])
            end
            log_alpha[t, j] = ASR::MathUtil.logsumexp(scores.to_a) + log_b[t, j]
          end
        end

        last = log_alpha[t_len - 1, true].to_a
        log_l = ASR::MathUtil.logsumexp(last)
        [log_alpha, log_l]
      end

      # Backward algorithm in log-domain.
      def backward(log_b)
        t_len = log_b.shape[0]
        log_beta = Numo::DFloat.zeros(t_len, @n_states) + LOG_ZERO

        # log(1) = 0 for final time step
        @n_states.times do |i|
          log_beta[t_len - 1, i] = 0.0
        end

        (t_len - 2).downto(0) do |t|
          @n_states.times do |i|
            scores = Numo::DFloat.zeros(@n_states) + LOG_ZERO
            @n_states.times do |j|
              next if @a[i, j] <= 0
              scores[j] = ::Math.log(@a[i, j]) + log_b[t + 1, j] + log_beta[t + 1, j]
            end
            log_beta[t, i] = ASR::MathUtil.logsumexp(scores.to_a)
          end
        end

        log_beta
      end

      # Posterior state and transition probabilities.
      # Returns [gamma (T,S), xi (T-1,S,S)]
      def posteriors(log_alpha, log_beta, log_b, log_l)
        t_len = log_alpha.shape[0]

        # gamma_ti ∝ exp(α_ti + β_ti - log_l)
        log_gamma = log_alpha + log_beta - log_l
        gamma = Numo::NMath.exp(log_gamma)

        xi = Numo::DFloat.zeros(t_len - 1, @n_states, @n_states)
        (0...(t_len - 1)).each do |t|
          @n_states.times do |i|
            @n_states.times do |j|
              next if @a[i, j] <= 0
              log_ijt = log_alpha[t, i] +
                        ::Math.log(@a[i, j]) +
                        log_b[t + 1, j] +
                        log_beta[t + 1, j] - log_l
              xi[t, i, j] = ::Math.exp(log_ijt)
            end
          end
        end

        [gamma, xi]
      end

      # Public helper: compute average log-likelihood over a list of sequences
      def score_sequences(sequences)
        total = 0.0
        sequences.each do |x|
          log_b = emission_log_likelihoods(x)
          _log_alpha, log_l = forward(log_b)
          total += log_l
        end
        total / sequences.length
      end
    end
  end
end
