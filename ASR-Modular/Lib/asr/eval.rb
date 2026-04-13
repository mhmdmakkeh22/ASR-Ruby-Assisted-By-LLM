# frozen_string_literal: true

require_relative "utils"

module ASR
  module Eval
    Result = Struct.new(:wer, :sub, :ins, :del, :ref_len)

    def self.edit_distance(ref, hyp)
      # Classic Levenshtein on word tokens
      r = ref.length
      h = hyp.length
      dp = Array.new(r + 1) { Array.new(h + 1, 0) }

      (0..r).each { |i| dp[i][0] = i }
      (0..h).each { |j| dp[0][j] = j }

      (1..r).each do |i|
        (1..h).each do |j|
          cost = (ref[i - 1] == hyp[j - 1]) ? 0 : 1
          dp[i][j] = [
            dp[i - 1][j] + 1,      # deletion
            dp[i][j - 1] + 1,      # insertion
            dp[i - 1][j - 1] + cost # substitution
          ].min
        end
      end

      # Backtrace to count S/I/D
      i = r
      j = h
      sub = ins = del = 0
      while i > 0 || j > 0
        if i > 0 && j > 0 && dp[i][j] == dp[i - 1][j - 1] && ref[i - 1] == hyp[j - 1]
          i -= 1
          j -= 1
        elsif i > 0 && j > 0 && dp[i][j] == dp[i - 1][j - 1] + 1
          sub += 1
          i -= 1
          j -= 1
        elsif i > 0 && dp[i][j] == dp[i - 1][j] + 1
          del += 1
          i -= 1
        else
          ins += 1
          j -= 1
        end
      end

      [sub, ins, del]
    end

    def self.wer(ref_words, hyp_words)
      sub, ins, del = edit_distance(ref_words, hyp_words)
      n = ref_words.length
      wer = (sub + ins + del) / [n, 1].max.to_f
      Result.new(wer, sub, ins, del, n)
    end
  end
end
