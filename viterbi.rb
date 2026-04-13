# frozen_string_literal: true

module ASR
  module Decoder
    class Viterbi
      # Decode a single WordHMM for observation sequence frames (T x D)
      # Returns best log-score and best path (state indices)
      def self.decode_word_hmm(word_hmm, frames)
        t_max = frames.length
        n = 5

        neg_inf = -1.0/0.0

        dp = Array.new(t_max) { Array.new(n, neg_inf) }
        bp = Array.new(t_max) { Array.new(n, nil) }

        # init: start at state 0
        dp[0][0] = word_hmm.log_emit(0, frames[0])
        (1...n).each { |s| dp[0][s] = neg_inf }

        # recursion
        (1...t_max).each do |t|
          (0...n).each do |s|
            emit = word_hmm.log_emit(s, frames[t])
            best_val = neg_inf
            best_prev = nil

            # allowed prev: s (self-loop) or s-1 (move)
            prevs = []
            prevs << s
            prevs << (s - 1) if s > 0

            prevs.each do |p|
              trans = word_hmm.log_a[p][s]
              next if trans.infinite? # -inf
              val = dp[t-1][p] + trans
              if val > best_val
                best_val = val
                best_prev = p
              end
            end

            dp[t][s] = best_val + emit
            bp[t][s] = best_prev
          end
        end

        # termination: must end in final state 4 (classic word-HMM)
        best_score = dp[t_max-1][4]
        path = Array.new(t_max)
        s = 4
        (t_max-1).downto(0) do |t|
          path[t] = s
          s = bp[t][s] || 0
        end

        { score: best_score, path: path }
      end
    end
  end
end
