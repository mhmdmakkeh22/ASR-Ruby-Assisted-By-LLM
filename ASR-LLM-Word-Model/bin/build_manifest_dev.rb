#!/usr/bin/env ruby
# frozen_string_literal: true

require "fileutils"
require "csv"

ROOT      = File.expand_path("..", __dir__)
DATA_ROOT = File.join(ROOT, "dataset", "dev-clean", "LibriSpeech")
OUT_DIR   = File.join(ROOT, "data", "manifests")
OUT_PATH  = File.join(OUT_DIR, "dev_manifest.tsv")

FileUtils.mkdir_p(OUT_DIR)

CSV.open(OUT_PATH, "w", col_sep: "\t") do |csv|
  csv << %w[id path text]

  Dir.glob(File.join(DATA_ROOT, "**", "*.trans.txt")).sort.each do |trans_file|
    dir = File.dirname(trans_file)

    File.foreach(trans_file) do |line|
      line = line.strip
      next if line.empty?

      utt_id, *words = line.split
      text = words.join(" ")

      # LibriSpeech audio file has same id + .flac
      audio_path = File.join(dir, "#{utt_id}.flac")

      csv << [utt_id, audio_path, text]
    end
  end
end

puts "Wrote manifest to #{OUT_PATH}"