# frozen_string_literal: true

require_relative "utils"

module ASR
  module Lexicon
    class LexiconStore
      attr_reader :word2phones

      def initialize(word2phones, g2p: :grapheme)
        @word2phones = word2phones # "WORD" => ["P1","P2",...]
        @g2p = g2p
      end

      def phones_for(word)
        w = word.to_s.upcase
        return ["SIL"] if w == "<SIL>" || w == "SIL" || w == "<SILENCE>"

        phones = @word2phones[w]
        return phones if phones

        # Fallback: grapheme "phones"
        ASR::Lexicon.g2p_grapheme(w)
      end

      def vocab
        @word2phones.keys.sort
      end

      def all_phones
        ps = @word2phones.values.flatten.uniq
        ps |= ["SIL"]
        ps.sort
      end
    end

    def self.load_tsv(path)
      m = {}
      ASR::Utils.read_tsv(path).each do |cols|
        next if cols.empty?
        word = cols[0].to_s.strip.upcase
        phones = cols[1].to_s.strip.split(/\s+/)
        next if word.empty? || phones.empty?
        m[word] = phones
      end
      LexiconStore.new(m)
    end

    def self.build_from_manifest(manifest_rows, out_path:)
      words = manifest_rows.flat_map { |r| ASR::Utils.tokenize(r["text"]) }.uniq.sort
      lines = []
      # Always include a silence word entry (not emitted as output word by default)
      lines << "<SIL>\tSIL"

      words.each do |w|
        phones = g2p_grapheme(w)
        lines << "#{w}\t#{phones.join(' ')}"
      end

      ASR::Utils.write_lines(out_path, lines)
      out_path
    end

    def self.g2p_grapheme(word)
      w = word.to_s.upcase.gsub(/[^A-Z0-9]/, "")
      return ["G_SP"] if w.empty?
      w.chars.map { |ch| "G_#{ch}" }
    end
  end
end
