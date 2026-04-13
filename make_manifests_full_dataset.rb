# frozen_string_literal: true
# Build train/test manifests directly from Speech Commands folder without copying files.
#
# Usage:
#   ruby bin/make_manifests_full_dataset.rb <SPEECH_COMMANDS_DATA_ROOT> <OUT_SPLITS_DIR> <WORDS_COMMA_LIST>
#
# Example:
#   ruby bin/make_manifests_full_dataset.rb "C:\Users\moham\Documents\ASR_LLM\Code\data\archive\data" data/splits yes,no,up,down,stop
#
# Output:
#   <OUT_SPLITS_DIR>/train_manifest.json
#   <OUT_SPLITS_DIR>/test_manifest.json
#
require "json"
require "fileutils"

def read_list(path)
  return nil unless File.exist?(path)
  File.readlines(path).map { |l| l.strip }.reject(&:empty?)
end

root = ARGV[0]
out_dir = ARGV[1]
words_csv = ARGV[2]
abort("Usage: ruby bin/make_manifests_full_dataset.rb <ROOT> <OUT_DIR> <words_csv>") if root.nil? || out_dir.nil? || words_csv.nil?
abort("Root folder does not exist: #{root}") unless Dir.exist?(root)

if words_csv == "ALL"
  words = Dir.children(root).select do |f|
    dir_path = File.join(root, f)
    File.directory?(dir_path) && !f.start_with?("_")
  end
  puts "Auto-detected #{words.length} word folders."
else
  words = words_csv.split(",").map(&:strip)
end
abort("No words provided") if words.empty?

FileUtils.mkdir_p(out_dir)

testing_list = read_list(File.join(root, "testing_list.txt"))
validation_list = read_list(File.join(root, "validation_list.txt"))

# Lists in Speech Commands are usually relative paths like: "yes/xxx.wav"
test_set = testing_list ? testing_list.to_h { |p| [p, true] } : {}
val_set  = validation_list ? validation_list.to_h { |p| [p, true] } : {}

entries_by_label = {}

words.each do |w|
  dir = File.join(root, w)
  abort("Missing word folder: #{dir}") unless Dir.exist?(dir)
  wavs = Dir.children(dir)
            .map { |f| File.join(dir, f) }
            .select { |p| File.file?(p) && File.extname(p).downcase == ".wav" }
            .sort

  abort("No wav files in: #{dir}") if wavs.empty?
  entries_by_label[w] = wavs
end

train = []
test = []

if testing_list && validation_list
  # Official split: test = testing_list; validation is excluded from train by default
  entries_by_label.each do |label, wavs|
    wavs.each do |abs_path|
      rel = "#{label}/#{File.basename(abs_path)}"
      if test_set[rel]
        test << { path: abs_path, label: label }
      elsif val_set[rel]
        # optional: you could create a val manifest; for now we skip validation
        next
      else
        train << { path: abs_path, label: label }
      end
    end
  end
  puts "Used official testing_list.txt and validation_list.txt."
else
  # Fallback: deterministic 80/20 split per label
  entries_by_label.each do |label, wavs|
    n = wavs.length
    n_test = (0.2 * n).floor
    n_test = 1 if n_test < 1
    # deterministic: last n_test for test
    wavs[0...(n - n_test)].each { |p| train << { path: p, label: label } }
    wavs[(n - n_test)..-1].each { |p| test << { path: p, label: label } }
  end
  puts "Official split files not found. Used fallback 80/20 split per label."
end

File.write(File.join(out_dir, "train_manifest.json"), JSON.pretty_generate(train))
File.write(File.join(out_dir, "test_manifest.json"), JSON.pretty_generate(test))

puts "Wrote: #{File.join(out_dir, 'train_manifest.json')} (#{train.length} items)"
puts "Wrote: #{File.join(out_dir, 'test_manifest.json')} (#{test.length} items)"
