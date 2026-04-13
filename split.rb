# frozen_string_literal: true

require "json"

module ASR
  module Dataset
    class Split
      # Deterministic per-label split: first n_train files → train, next n_test → test
      def self.fixed_per_label(entries, n_train:, n_test:)
        by_label = entries.group_by(&:label)
        train = []
        test  = []

        by_label.each do |label, list|
          list_sorted = list.sort_by(&:path)
          raise "Not enough files for label=#{label}. Need #{n_train + n_test}, have #{list_sorted.size}" if list_sorted.size < (n_train + n_test)
          train.concat(list_sorted[0, n_train])
          test.concat(list_sorted[n_train, n_test])
        end

        { train: train, test: test }
      end

      def self.save(split_hash, out_dir)
        Dir.mkdir(out_dir) unless Dir.exist?(out_dir)
        File.write(File.join(out_dir, "train_manifest.json"), JSON.pretty_generate(split_hash[:train].map { |e| { path: e.path, label: e.label } }))
        File.write(File.join(out_dir, "test_manifest.json"),  JSON.pretty_generate(split_hash[:test].map  { |e| { path: e.path, label: e.label } }))
      end
    end
  end
end
