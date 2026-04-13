# frozen_string_literal: true

module ASR
  module Lexicon
    class Lexicon
      def initialize(map)
        @map = map # word_upcase -> model_id (string)
      end

      def self.load(path)
        map = {}
        File.readlines(path).each do |line|
          line = line.strip
          next if line.empty? || line.start_with?("#")
          parts = line.split(/\s+/)
          raise "Bad lexicon line: #{line}" if parts.size < 2
          word = parts[0].upcase
          model_id = parts[1]
          map[word] = model_id
        end
        new(map)
      end

      def words
        @map.keys
      end

      def model_id_for(word)
        @map.fetch(word.upcase)
      end
    end
  end
end

