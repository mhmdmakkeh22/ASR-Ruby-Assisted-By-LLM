# frozen_string_literal: true

require "json"

module ASR
  module Dataset
    Entry = Struct.new(:path, :label, keyword_init: true)

    class Manifest
      # Scan data/words/<label>/*.wav
      def self.scan(root_dir)
        entries = []
        Dir.glob(File.join(root_dir, "*")).select { |p| File.directory?(p) }.each do |label_dir|
          label = File.basename(label_dir)
          Dir.glob(File.join(label_dir, "*.wav")).sort.each do |wav|
            entries << Entry.new(path: wav, label: label)
          end
        end
        entries
      end

      def self.to_json(entries)
        JSON.pretty_generate(entries.map { |e| { path: e.path, label: e.label } })
      end

      def self.from_json(path)
        arr = JSON.parse(File.read(path))
        arr.map { |h| Entry.new(path: h["path"], label: h["label"]) }
      end
    end
  end
end

