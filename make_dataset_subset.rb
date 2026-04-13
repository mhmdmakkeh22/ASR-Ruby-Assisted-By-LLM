# frozen_string_literal: true
# Usage:
#   ruby bin/make_dataset_subset.rb <SPEECH_COMMANDS_ROOT> <OUT_DIR>
#
# Example:
#   ruby bin/make_dataset_subset.rb C:\datasets\speech_commands_v0.02 data/words
#
# It copies a SMALL subset for 5 words: yes no up down stop
# Then creates train/test manifests: data/splits/train_manifest.json and test_manifest.json

require "fileutils"
require_relative "../lib/asr/dataset/manifest"
require_relative "../lib/asr/dataset/split"

src = ARGV[0]
out = ARGV[1]
abort("Usage: ruby bin/make_dataset_subset.rb <SPEECH_COMMANDS_ROOT> <OUT_DIR>") if src.nil? || out.nil?

words = %w[yes no up down stop]
n_train = 20
n_test  = 5

FileUtils.mkdir_p(out)

words.each do |w|
  src_dir = File.join(src, w)
  abort("Missing folder: #{src_dir}") unless Dir.exist?(src_dir)
  dst_dir = File.join(out, w)
  FileUtils.mkdir_p(dst_dir)

  files = Dir.glob(File.join(src_dir, "*.wav")).sort
  abort("Not enough files in #{src_dir}") if files.size < (n_train + n_test)

  # copy first n_train+n_test deterministically
  files[0, n_train + n_test].each_with_index do |f, idx|
    FileUtils.cp(f, File.join(dst_dir, "#{w}_#{format('%03d', idx+1)}.wav"))
  end
end

entries = ASR::Dataset::Manifest.scan(out)
split = ASR::Dataset::Split.fixed_per_label(entries, n_train: n_train, n_test: n_test)
ASR::Dataset::Split.save(split, "data/splits")

puts "Done."
puts "Copied subset to: #{out}"
puts "Wrote manifests to: data/splits/train_manifest.json and data/splits/test_manifest.json"
