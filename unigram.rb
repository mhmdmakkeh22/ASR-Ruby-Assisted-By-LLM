FILE: lib/asr/lm/unigram.rb

# frozen_string_literal: true

require "json"

module ASR
  module LM
    class Unigram
      def initialize(probs)
        @probs = probs # word_upcase -> prob
        @logp  = {}
        @probs.each { |w, p| @logp[w] = ::Math.log([p, 1e-12].max) }
      end

      def self.load(path)
        h = JSON.parse(File.read(path))
        raise "LM type must be unigram" unless h["type"] == "unigram"
        probs = {}
        h["probs"].each { |k, v| probs[k.upcase] = v.to_f }
        new(probs)
      end

      def logp(word)
        @logp.fetch(word.upcase, ::Math.log(1e-12))
      end
    end
  end
end

