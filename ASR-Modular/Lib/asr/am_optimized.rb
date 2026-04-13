# frozen_string_literal: true

require_relative "am"

module ASR
  module AM
    class OptimizedHMMGMM < HMMGMM
      def initialize(n_states:, n_components:, dim:, var_floor: 1e-3)
        super(n_states: n_states, n_components: n_components, dim: dim, var_floor: var_floor)
        @emit_cache = {} # Cache for log_emit calculations
        @cache_hits = 0
        @cache_misses = 0
      end

      # Optimized training with multiple speed improvements
      def train!(utterances, lexicon, n_iter: 8, trainer: :bw, insert_sil: true, 
                 verbose: true, subset_ratio: 1.0, early_pruning: true)
        raise ArgumentError, "utterances empty" if utterances.empty?
        
        # Use subset for faster training
        if subset_ratio < 1.0
          subset_size = (utterances.length * subset_ratio).to_i
          utterances = utterances.sample(subset_size)
          puts "Using subset: #{subset_size}/#{utterances.length + subset_size}" if verbose
        end
        
        flat_start!(utterances, lexicon, insert_sil: insert_sil) if @phones.empty?
        
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
            if verbose && (idx % 100 == 0 || idx == utterances.length - 1)
              progress = (idx + 1) * 100.0 / utterances.length
              puts "  Progress: #{progress.round(1)}% (#{idx + 1}/#{utterances.length}) - Processing #{u[:utt_id]}"
            end
            
            phones = phones_for_words(u[:words], lexicon, insert_sil: insert_sil)
            next if phones.empty?
            next unless phones.all? { |p| @phones.key?(p) }
            feats = u[:feats]
            next if feats.empty?
            
            ll = if trainer.to_sym == :viterbi
              viterbi_accumulate_optimized!(phones, feats, gmm_stats, early_pruning)
            else
              baum_welch_accumulate_optimized!(phones, feats, gmm_stats, early_pruning)
            end
            
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
          puts format("Iter %02d/%02d avg loglik=%.3f used_utts=%d cache_hit_rate=%.2f%% time=%.2fs", 
                     it + 1, n_iter, avg, used, cache_hit_rate, iter_time) if verbose
          
          # Show model statistics
          if verbose
            phone_count = @phones.length
            total_states = phone_count * @n_states
            total_components = total_states * @n_components
            puts "  Model: #{phone_count} phones, #{total_states} states, #{total_components} GMM components"
            puts "  Cache: #{@cache_hits} hits, #{@cache_misses} misses"
          end
          
          # Clear cache between iterations to prevent memory bloat
          clear_emit_cache! if it < n_iter - 1
        end
      end

      # Memory-efficient Baum-Welch with pruning
      def baum_welch_accumulate_optimized!(phones, feats, stats, early_pruning = true)
        s_total = phones.length * @n_states
        t_total = feats.length
        return 0.0 if t_total < s_total

        # Pruning threshold - keep only likely states
        prune_thresh = early_pruning ? -20.0 : -Float::INFINITY
        
        # Forward pass - compute alpha on the fly
        alpha_prev = Array.new(s_total, -Float::INFINITY)
        alpha_prev[0] = log_emit_cached(phones[0], 0, feats[0])
        
        total_loglik = 0.0
        
        (1...t_total).each do |t|
          x = feats[t]
          alpha_curr = Array.new(s_total, -Float::INFINITY)
          
          s_total.times do |s|
            ph = phones[s / @n_states]
            st = s % @n_states
            
            # Self transition
            a1 = alpha_prev[s] + log_self(ph, st)
            
            # Next transition from previous state
            a2 = -Float::INFINITY
            if s > 0
              prev_ph = phones[(s - 1) / @n_states]
              prev_st = (s - 1) % @n_states
              a2 = alpha_prev[s - 1] + log_next(prev_ph, prev_st)
            end
            
            alpha_curr[s] = log_emit_cached(ph, st, x) + ASR::Utils.log_sum_exp(a1, a2)
            
            # Apply pruning
            alpha_curr[s] = prune_thresh if alpha_curr[s] < prune_thresh
          end
          
          alpha_prev = alpha_curr
        end
        
        # Final likelihood
        last_ph = phones[(s_total - 1) / @n_states]
        last_st = (s_total - 1) % @n_states
        loglik = alpha_prev[s_total - 1] + log_next(last_ph, last_st)
        
        # Backward pass and accumulation in one go
        beta_next = Array.new(s_total, -Float::INFINITY)
        beta_next[s_total - 1] = log_next(last_ph, last_st)
        
        (t_total - 2).downto(0) do |t|
          x = feats[t]
          beta_curr = Array.new(s_total, -Float::INFINITY)
          
          s_total.times do |s|
            ph = phones[s / @n_states]
            st = s % @n_states
            
            # Accumulate stats for current frame
            if t > 0 # Skip first frame for proper gamma calculation
              gamma = Math.exp(alpha_prev[s] + beta_next[s] - loglik)
              next unless gamma > 1e-10 # Skip very small contributions
              
              key = "#{ph}\t#{st}"
              accumulate_gmm_stats!(key, x, gamma, stats)
            end
            
            # Beta computation
            b1 = log_self(ph, st) + log_emit_cached(ph, st, feats[t + 1]) + beta_next[s]
            b2 = -Float::INFINITY
            if s < s_total - 1
              b2 = log_next(ph, st) + log_emit_cached(phones[s + 1] / @n_states, (s + 1) % @n_states, feats[t + 1]) + beta_next[s + 1]
            end
            beta_curr[s] = ASR::Utils.log_sum_exp(b1, b2)
          end
          
          beta_next = beta_curr
        end
        
        loglik
      end

      # Optimized Viterbi with pruning
      def viterbi_accumulate_optimized!(phones, feats, stats, early_pruning = true)
        s_total = phones.length * @n_states
        t_total = feats.length
        return 0.0 if t_total < s_total

        prune_thresh = early_pruning ? -15.0 : -Float::INFINITY
        
        dp_prev = Array.new(s_total, -Float::INFINITY)
        dp_prev[0] = log_emit_cached(phones[0], 0, feats[0])
        
        backpointer = []
        
        (1...t_total).each do |t|
          x = feats[t]
          dp_curr = Array.new(s_total, -Float::INFINITY)
          bp_curr = Array.new(s_total, 0)
          
          s_total.times do |s|
            ph = phones[s / @n_states]
            st = s % @n_states
            
            # Self transition
            best = dp_prev[s] + log_self(ph, st)
            arg = s
            
            # Next transition
            if s > 0
              prev_ph = phones[(s - 1) / @n_states]
              prev_st = (s - 1) % @n_states
              cand = dp_prev[s - 1] + log_next(prev_ph, prev_st)
              if cand > best
                best = cand
                arg = s - 1
              end
            end
            
            dp_curr[s] = best + log_emit_cached(ph, st, x)
            bp_curr[s] = arg
            
            # Apply pruning
            dp_curr[s] = prune_thresh if dp_curr[s] < prune_thresh
          end
          
          dp_prev = dp_curr
          backpointer << bp_curr
        end
        
        # Traceback and accumulate stats
        last_ph = phones[(s_total - 1) / @n_states]
        last_st = (s_total - 1) % @n_states
        loglik = dp_prev[s_total - 1] + log_next(last_ph, last_st)
        
        # Reconstruct path
        path = Array.new(t_total, 0)
        s = s_total - 1
        (t_total - 1).downto(0) do |t|
          path[t] = s
          s = t > 0 ? backpointer[t - 1][s] : 0
        end
        
        # Accumulate statistics along Viterbi path
        t_total.times do |t|
          s = path[t]
          ph = phones[s / @n_states]
          st = s % @n_states
          key = "#{ph}\t#{st}"
          x = feats[t]
          accumulate_gmm_stats!(key, x, 1.0, stats)
        end
        
        loglik
      end

      private

      def log_emit_cached(phone, state_i, x)
        key = "#{phone}\t#{state_i}\t#{x.object_id}"
        
        if @emit_cache.key?(key)
          @cache_hits += 1
          return @emit_cache[key]
        end
        
        @cache_misses += 1
        result = log_emit(phone, state_i, x)
        
        # Limit cache size to prevent memory bloat
        if @emit_cache.size > 10000
          @emit_cache.clear
        end
        
        @emit_cache[key] = result
        result
      end

      def accumulate_gmm_stats!(key, x, gamma, stats)
        comp_logs = @phones[key.split("\t").first].gmms[key.split("\t").last.to_i].log_component_pdfs(x)
        lse = ASR::Utils.logsumexp(comp_logs)
        
        @n_components.times do |k|
          r = Math.exp(comp_logs[k] - lse)
          weight = gamma * r
          
          stats[key][:n_k][k] += weight
          @dim.times do |d|
            xd = x[d]
            stats[key][:f_kd][k][d] += weight * xd
            stats[key][:s_kd][k][d] += weight * xd * xd
          end
        end
      end

      def clear_emit_cache!
        cache_size = @emit_cache.size
        @emit_cache.clear
        puts "  Cache cleared: #{cache_size} entries removed" if cache_size > 0
      end

      def cache_hit_rate
        total = @cache_hits + @cache_misses
        return 0.0 if total == 0
        (@cache_hits.to_f / total * 100.0).round(2)
      end
    end
  end
end
