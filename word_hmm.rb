# frozen_string_literal: true

require "json"
require_relative "gmm"

module ASR
  module Model
    class WordHMM
      # 5-state left-to-right HMM (emitting states only)
      # transitions: i->i and i->i+1 ; state 5 is absorbing
      attr_reader :word, :states, :log_a

      def initialize(word:, state_gmms:, a_stay: 0.6, a_move: 0.4)
        @word = word.upcase
        raise "Need 5 state gmms" unless state_gmms.length == 5
        @states = state_gmms # Array[GMM] length 5

        # log transition matrix 5x5 with -inf for impossible transitions
        @log_a = Array.new(5) { Array.new(5, -1.0/0.0) } # -Infinity
        (0..3).each do |i|
          @log_a[i][i] = ::Math.log(a_stay)
          @log_a[i][i+1] = ::Math.log(a_move)
        end
        @log_a[4][4] = 0.0
      end

      def log_emit(state_idx, x_t)
        @states[state_idx].logpdf(x_t)
      end

      def to_h
        {
          word: @word,
          a: @log_a,
          states: @states.map(&:to_h)
        }
      end

      def self.from_h(h)
        word = h["word"]
        a = h["a"]
        gmms = h["states"].map { |sh| GMM.from_h(sh) }
        obj = allocate
        obj.instance_variable_set(:@word, word)
        obj.instance_variable_set(:@states, gmms)
        obj.instance_variable_set(:@log_a, a)
        obj
      end
    end
  end
end
