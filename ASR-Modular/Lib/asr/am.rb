# lib/asr/am.rb
# frozen_string_literal: true
require_relative "utils"
module ASR
  module AM
    # Diagonal-covariance Gaussian Mixture Model:
    # p(x) = sum_k w_k * N(x | mu_k, diag(var_k))
    class DiagGMM
      attr_reader :k, :dim, :w, :mu, :var

      def initialize(w, mu, var)
        @w = w
        @mu = mu
        @var = var
        @k = w.length
        @dim = mu.first.length
        recompute_cache!
      end

      def self.from_data(data, k:, var_floor: 1e-3)
        raise ArgumentError, "empty data" if data.empty?
        dim = data.first.length
        mean = Array.new(dim, 0.0)
        data.each { |x| dim.times { |d| mean[d] += x[d] } }
        dim.times { |d| mean[d] /= data.length.to_f }
        var = Array.new(dim, 0.0)
        data.each do |x|
          dim.times do |d|
            diff = x[d] - mean[d]
            var[d] += diff * diff
          end
        end
        dim.times do |d|
          var[d] = var[d] / [data.length - 1, 1].max
          var[d] = var_floor if var[d] < var_floor
        end
        picks = data.sample([k, data.length].min)
        mu = Array.new(k) { |i| picks[i % picks.length].dup }
        vv = Array.new(k) { var.dup }
        w = Array.new(k, 1.0 / k.to_f)
        new(w, mu, vv)
      end

      def recompute_cache!
        @log_w = @w.map { |a| Math.log([a, ASR::Utils::EPS].max) }
        @inv_var = @var.map { |v| v.map { |x| 1.0 / x } }
        @log_norm = @var.map do |v|
          s = 0.0
          v.each { |x| s += Math.log(2.0 * Math::PI * x) }
          -0.5 * s
        end
      end

      def log_component_pdfs(x)
        out = Array.new(@k, 0.0)
        @k.times do |i|
          s = 0.0
          @dim.times do |d|
            diff = x[d] - @mu[i][d]
            s += diff * diff * @inv_var[i][d]
          end
          out[i] = @log_w[i] + @log_norm[i] - 0.5 * s
        end
        out
      end

      def log_pdf(x)
        ASR::Utils.logsumexp(log_component_pdfs(x))
      end

      # Update from sufficient statistics:
      # N_k = sum_t gamma_tk
      # F_kd = sum_t gamma_tk * x_td
      # S_kd = sum_t gamma_tk * x_td^2
      def update_from_stats!(n_k, f_kd, s_kd, var_floor: 1e-3, weight_floor: 1e-6)
        tot = n_k.sum.to_f
        return if tot <= 0.0
        @k.times do |i|
          @w[i] = [n_k[i] / tot, weight_floor].max
        end
        z = @w.sum.to_f
        @k.times { |i| @w[i] /= z }
        @k.times do |i|
          next if n_k[i] <= 0.0
          @dim.times do |d|
            mu = f_kd[i][d] / n_k[i]
            ex2 = s_kd[i][d] / n_k[i]
            vv = ex2 - mu * mu
            vv = var_floor if vv < var_floor
            @mu[i][d] = mu
            @var[i][d] = vv
          end
        end
        recompute_cache!
      end
    end

    PhoneModel = Struct.new(:self_probs, :gmms)

    class HMMGMM
      attr_reader :n_states, :n_components, :dim, :phones

      def initialize(n_states:, n_components:, dim:, var_floor: 1e-3)
        @n_states = n_states
        @n_components = n_components
        @dim = dim
        @var_floor = var_floor
        @phones = {} # "PHONE" => PhoneModel
      end

      # CHANGE: helper for decoder to check inventory without raising.
      def has_phone?(phone)
        @phones.key?(phone.to_s)
      end

      # CHANGE: safe emit for decoder; returns -inf instead of raising on missing phone/state.
      def log_emit_safe(phone, state_i, x)
        pm = @phones[phone.to_s]
        return -Float::INFINITY unless pm
        gmm = pm.gmms[state_i] rescue nil
        return -Float::INFINITY unless gmm
        gmm.log_pdf(x)
      end

      # CHANGE: safe transition helpers for decoder; avoid fetch on missing phones.
      def log_self_safe(phone, state_i)
        pm = @phones[phone.to_s]
        return -Float::INFINITY unless pm
        p = pm.self_probs[state_i] rescue nil
        return -Float::INFINITY unless p
        Math.log([p, ASR::Utils::EPS].max)
      end

      def log_next_safe(phone, state_i)
        pm = @phones[phone.to_s]
        return -Float::INFINITY unless pm
        pself = pm.self_probs[state_i] rescue nil
        return -Float::INFINITY unless pself
        Math.log([1.0 - pself, ASR::Utils::EPS].max)
      end

      def ensure_phone!(phone, seed_data: nil)
        p = phone.to_s
        return if @phones.key?(p)
        raise ArgumentError, "need seed_data to init phone=#{p}" if seed_data.nil? || seed_data.empty?
        gmms = Array.new(@n_states) { DiagGMM.from_data(seed_data, k: @n_components, var_floor: @var_floor) }
        self_probs = Array.new(@n_states, 0.6)
        self_probs[-1] = 0.7
        @phones[p] = PhoneModel.new(self_probs, gmms)
      end

      # STRICT (training): keep fetch-based methods to surface config/model bugs early.
      def log_self(phone, state_i)
        Math.log([@phones.fetch(phone).self_probs[state_i], ASR::Utils::EPS].max)
      end

      def log_next(phone, state_i)
        Math.log([1.0 - @phones.fetch(phone).self_probs[state_i], ASR::Utils::EPS].max)
      end

      def log_emit(phone, state_i, x)
        @phones.fetch(phone).gmms[state_i].log_pdf(x)
      end

      def phones_for_words(words, lexicon, insert_sil: true)
        seq = []
        seq << "SIL" if insert_sil
        words.each do |w|
          seq.concat(lexicon.phones_for(w))
          seq << "SIL" if insert_sil
        end
        seq
      end

      def flat_start!(utterances, lexicon, insert_sil: true)
        frames_by_key = Hash.new { |h, k| h[k] = [] }
        all_frames = []
        utterances.each { |u| all_frames.concat(u[:feats]) }
        raise "No frames to initialize" if all_frames.empty?
        ensure_phone!("SIL", seed_data: all_frames)
        utterances.each do |u|
          feats = u[:feats]
          phones = phones_for_words(u[:words], lexicon, insert_sil: insert_sil)
          s_total = phones.length * @n_states
          t_total = feats.length
          next if s_total <= 0 || t_total <= 0
          next if t_total < s_total
          (0...s_total).each do |s|
            t0 = (s * t_total / s_total.to_f).floor
            t1 = (((s + 1) * t_total / s_total.to_f).floor - 1)
            t1 = [t1, t_total - 1].min
            phone = phones[s / @n_states]
            st = s % @n_states
            key = "#{phone}\t#{st}"
            (t0..t1).each { |t| frames_by_key[key] << feats[t] } if t0 <= t1
          end
        end
        phone_set = frames_by_key.keys.map { |k| k.split("\t").first }.uniq
        phone_set |= ["SIL"]
        phone_set.each { |ph| ensure_phone!(ph, seed_data: all_frames) unless @phones.key?(ph) }
        @phones.each do |phone, pm|
          @n_states.times do |st|
            key = "#{phone}\t#{st}"
            data = frames_by_key[key]
            data = all_frames if data.nil? || data.empty?
            pm.gmms[st] = DiagGMM.from_data(data, k: @n_components, var_floor: @var_floor)
          end
        end
      end

      def train!(utterances, lexicon, n_iter: 8, trainer: :bw, insert_sil: true, verbose: true)
        raise ArgumentError, "utterances empty" if utterances.empty?
        
        puts "Starting flat-start initialization..." if verbose
        flat_start!(utterances, lexicon, insert_sil: insert_sil) if @phones.empty?
        puts "Initialization complete: #{@phones.length} phones" if verbose
        
        n_iter.times do |it|
          iter_start_time = Time.now
          puts "Starting iteration #{it + 1}/#{n_iter}..." if verbose
          
          gmm_stats = Hash.new do |h, key|
            h[key] = {
              n_k: Array.new(@n_components, 0.0),
              f_kd: Array.new(@n_components) { Array.new(@dim, 0.0) },
              s_kd: Array.new(@n_components) { Array.new(@dim, 0.0) },
              trans_self: 0.0,
              trans_next: 0.0
            }
          end
          
          total_loglik = 0.0
          used = 0
          utterance_start_time = Time.now
          utterances.each_with_index do |u, idx|
            if verbose && (idx % 200 == 0 || idx == utterances.length - 1)
              progress = (idx + 1) * 100.0 / utterances.length
              puts "  Progress: #{progress.round(1)}% (#{idx + 1}/#{utterances.length}) - Processing #{u[:utt_id]}"
            end
            
            phones = phones_for_words(u[:words], lexicon, insert_sil: insert_sil)
            next if phones.empty?
            next unless phones.all? { |p| @phones.key?(p) }
            feats = u[:feats]
            next if feats.empty?
            ll = (trainer.to_sym == :viterbi) ?
              viterbi_accumulate!(phones, feats, gmm_stats) :
              baum_welch_accumulate!(phones, feats, gmm_stats)
            total_loglik += ll
            used += 1
          end
          
          utterance_time = Time.now - utterance_start_time
          puts "  Utterance processing: #{utterance_time.round(2)}s" if verbose
          
          update_start_time = Time.now
          gmm_stats.each do |key, st|
            phone, state_i_s = key.split("\t")
            state_i = state_i_s.to_i
            pm = @phones[phone]
            denom = st[:trans_self] + st[:trans_next]
            if denom > 0.0
              pself = st[:trans_self] / denom
              pself = [[pself, 0.05].max, 0.95].min
              pm.self_probs[state_i] = pself
            end
            pm.gmms[state_i].update_from_stats!(st[:n_k], st[:f_kd], st[:s_kd], var_floor: @var_floor)
          end
          update_time = Time.now - update_start_time
          puts "  Model update: #{update_time.round(2)}s" if verbose
          
          avg = used > 0 ? total_loglik / used.to_f : -Float::INFINITY
          iter_time = Time.now - iter_start_time
          puts format("Iter %02d/%02d avg loglik=%.3f used_utts=%d time=%.2fs", 
                     it + 1, n_iter, avg, used, iter_time) if verbose
        end
      end

      # --- TRAINING CORE ---
      def baum_welch_accumulate!(phones, feats, stats)
        s_total = phones.length * @n_states
        t_total = feats.length
        return 0.0 if t_total < s_total
        log_b = Array.new(t_total) { Array.new(s_total, 0.0) }
        t_total.times do |t|
          x = feats[t]
          s_total.times do |s|
            ph = phones[s / @n_states]
            st = s % @n_states
            log_b[t][s] = log_emit(ph, st, x)
          end
        end
        alpha = Array.new(t_total) { Array.new(s_total, -Float::INFINITY) }
        alpha[0][0] = log_b[0][0]
        (1...t_total).each do |t|
          s_total.times do |s|
            a1 = alpha[t - 1][s] + log_self(phones[s / @n_states], s % @n_states)
            a2 = -Float::INFINITY
            if s > 0
              prev_ph = phones[(s - 1) / @n_states]
              prev_st = (s - 1) % @n_states
              a2 = alpha[t - 1][s - 1] + log_next(prev_ph, prev_st)
            end
            alpha[t][s] = log_b[t][s] + ASR::Utils.log_sum_exp(a1, a2)
          end
        end
        last_ph = phones[(s_total - 1) / @n_states]
        last_st = (s_total - 1) % @n_states
        loglik = alpha[t_total - 1][s_total - 1] + log_next(last_ph, last_st)
        beta = Array.new(t_total) { Array.new(s_total, -Float::INFINITY) }
        beta[t_total - 1][s_total - 1] = log_next(last_ph, last_st)
        (t_total - 2).downto(0) do |t|
          s_total.times do |s|
            ph = phones[s / @n_states]
            st = s % @n_states
            b1 = log_self(ph, st) + log_b[t + 1][s] + beta[t + 1][s]
            b2 = -Float::INFINITY
            if s < s_total - 1
              b2 = log_next(ph, st) + log_b[t + 1][s + 1] + beta[t + 1][s + 1]
            end
            beta[t][s] = ASR::Utils.log_sum_exp(b1, b2)
          end
        end
        t_total.times do |t|
          x = feats[t]
          s_total.times do |s|
            ph = phones[s / @n_states]
            st = s % @n_states
            key = "#{ph}\t#{st}"
            g = Math.exp(alpha[t][s] + beta[t][s] - loglik)
            next if g <= 0.0
            comp_logs = @phones[ph].gmms[st].log_component_pdfs(x)
            lse = ASR::Utils.logsumexp(comp_logs)
            @n_components.times do |k|
              r = Math.exp(comp_logs[k] - lse)
              w = g * r
              stats[key][:n_k][k] += w
              @dim.times do |d|
                xd = x[d]
                stats[key][:f_kd][k][d] += w * xd
                stats[key][:s_kd][k][d] += w * xd * xd
              end
            end
          end
        end
        (0...(t_total - 1)).each do |t|
          (0...s_total).each do |s|
            ph = phones[s / @n_states]
            st = s % @n_states
            key = "#{ph}\t#{st}"
            xi_self = alpha[t][s] + log_self(ph, st) + log_b[t + 1][s] + beta[t + 1][s] - loglik
            stats[key][:trans_self] += Math.exp(xi_self)
            if s < s_total - 1
              xi_next = alpha[t][s] + log_next(ph, st) + log_b[t + 1][s + 1] + beta[t + 1][s + 1] - loglik
              stats[key][:trans_next] += Math.exp(xi_next)
            end
          end
        end
        stats["#{last_ph}\t#{last_st}"][:trans_next] += 1.0
        loglik
      end

      def viterbi_accumulate!(phones, feats, stats)
        s_total = phones.length * @n_states
        t_total = feats.length
        return 0.0 if t_total < s_total
        dp = Array.new(t_total) { Array.new(s_total, -Float::INFINITY) }
        bp = Array.new(t_total) { Array.new(s_total, 0) }
        dp[0][0] = log_emit(phones[0], 0, feats[0])
        (1...t_total).each do |t|
          x = feats[t]
          (0...s_total).each do |s|
            ph = phones[s / @n_states]
            st = s % @n_states
            best = dp[t - 1][s] + log_self(ph, st)
            arg = s
            if s > 0
              prev_ph = phones[(s - 1) / @n_states]
              prev_st = (s - 1) % @n_states
              cand = dp[t - 1][s - 1] + log_next(prev_ph, prev_st)
              if cand > best
                best = cand
                arg = s - 1
              end
            end
            dp[t][s] = best + log_emit(ph, st, x)
            bp[t][s] = arg
          end
        end
        last_ph = phones[(s_total - 1) / @n_states]
        last_st = (s_total - 1) % @n_states
        ll = dp[t_total - 1][s_total - 1] + log_next(last_ph, last_st)
        path = Array.new(t_total, 0)
        s = s_total - 1
        (t_total - 1).downto(0) do |t|
          path[t] = s
          s = bp[t][s] if t > 0
        end
        t_total.times do |t|
          s = path[t]
          ph = phones[s / @n_states]
          st = s % @n_states
          key = "#{ph}\t#{st}"
          x = feats[t]
          comp_logs = @phones[ph].gmms[st].log_component_pdfs(x)
          lse = ASR::Utils.logsumexp(comp_logs)
          @n_components.times do |k|
            r = Math.exp(comp_logs[k] - lse)
            stats[key][:n_k][k] += r
            @dim.times do |d|
              xd = x[d]
              stats[key][:f_kd][k][d] += r * xd
              stats[key][:s_kd][k][d] += r * xd * xd
            end
          end
        end
        (0...(t_total - 1)).each do |t|
          s0 = path[t]
          s1 = path[t + 1]
          ph = phones[s0 / @n_states]
          st = s0 % @n_states
          key = "#{ph}\t#{st}"
          if s1 == s0
            stats[key][:trans_self] += 1.0
          else
            stats[key][:trans_next] += 1.0
          end
        end
        stats["#{last_ph}\t#{last_st}"][:trans_next] += 1.0
        ll
      end
    end
  end
end