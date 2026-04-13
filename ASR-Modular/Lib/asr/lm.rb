# frozen_string_literal: true

require "json"
require_relative "utils"

module ASR
  module LM
    class Unigram
      def initialize(logp, logp_unk)
        @logp = logp # "WORD" => logprob
        @logp_unk = logp_unk
      end

      def log_prob(word)
        @logp.fetch(word.to_s.upcase, @logp_unk)
      end

      def vocab
        @logp.keys
      end

      def to_h
        { "logp" => @logp, "logp_unk" => @logp_unk }
      end

      def self.from_h(h)
        new(h.fetch("logp"), h.fetch("logp_unk"))
      end
    end

    def self.build_unigram(texts, add_one: 1.0)
      counts = Hash.new(0)
      total = 0
      texts.each do |txt|
        ASR::Utils.tokenize(txt).each do |w|
          counts[w] += 1
          total += 1
        end
      end
      vocab = counts.keys
      v = vocab.length

      # Add-one (Laplace) smoothing for a minimal LM.
      logp = {}
      denom = total + add_one * v
      vocab.each do |w|
        p = (counts[w] + add_one) / denom.to_f
        logp[w] = Math.log(p)
      end

      # Unknown word probability: treat as add_one / denom (simple fallback)
      logp_unk = Math.log(add_one / denom.to_f)

      Unigram.new(logp, logp_unk)
    end

    def self.save_json(path, lm)
      ASR::Utils.ensure_dir(File.dirname(path))
      File.write(path, JSON.pretty_generate(lm.to_h))
      path
    end

    def self.load_json(path)
      Unigram.from_h(JSON.parse(File.read(path)))
    end
  end
end
