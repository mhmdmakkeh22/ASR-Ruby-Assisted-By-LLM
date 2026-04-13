#!/usr/bin/env ruby
# frozen_string_literal: true

require "json"

train_manifest = "Data/manifests/train_clean_100.jsonl"
subset_size = ARGV[0]&.to_i || 5000
output_dir = "Data/manifests/balanced_#{subset_size}"

puts "Creating balanced dataset from train-clean-100..."
puts "Target subset size: #{subset_size}"
puts "Output directory: #{output_dir}"

# Create output directory
Dir.mkdir(output_dir) unless Dir.exist?(output_dir)

# Load all train utterances
train_utterances = []
File.readlines(train_manifest).each do |line|
  train_utterances << JSON.parse(line)
end

puts "Loaded #{train_utterances.length} utterances from train-clean-100"

# Group by speaker
speaker_utterances = {}
train_utterances.each do |utt|
  speaker = utt['id'].split('-')[0]
  speaker_utterances[speaker] ||= []
  speaker_utterances[speaker] << utt
end

puts "Found #{speaker_utterances.keys.length} speakers"

# Select speakers with enough data (at least 15 utterances)
qualified_speakers = speaker_utterances.select { |speaker, utts| utts.length >= 15 }
puts "Speakers with >=15 utterances: #{qualified_speakers.keys.length}"

# Select 20 speakers for balanced split
selected_speakers = qualified_speakers.keys.take(20)
puts "Selected speakers: #{selected_speakers.join(', ')}"

# Create balanced splits (70% train, 15% dev, 15% test)
train_split = []
dev_split = []
test_split = []

selected_speakers.each do |speaker|
  utts = qualified_speakers[speaker].shuffle
  
  # Take up to 250 utterances per speaker to keep it balanced
  utts = utts.take(250)
  
  # Split: 70% train, 15% dev, 15% test
  n_train = (utts.length * 0.7).floor
  n_dev = ((utts.length - n_train) / 2.0).floor
  n_test = utts.length - n_train - n_dev
  
  train_split.concat(utts[0...n_train])
  dev_split.concat(utts[n_train...(n_train + n_dev)])
  test_split.concat(utts[(n_train + n_dev)..-1])
end

puts "\nFinal split sizes:"
puts "Train: #{train_split.length} utterances"
puts "Dev: #{dev_split.length} utterances"  
puts "Test: #{test_split.length} utterances"
puts "Total: #{train_split.length + dev_split.length + test_split.length}"

# Write manifests
def write_manifest(utterances, filename)
  File.open(filename, "w") do |f|
    utterances.each { |utt| f.puts(JSON.generate(utt)) }
  end
  puts "Wrote #{utterances.length} utterances to #{filename}"
end

write_manifest(train_split, "#{output_dir}/train.jsonl")
write_manifest(dev_split, "#{output_dir}/dev.jsonl")
write_manifest(test_split, "#{output_dir}/test.jsonl")

# Analyze splits
def analyze_split(utterances, name)
  speakers = utterances.map { |utt| utt['id'].split('-')[0] }.uniq
  word_counts = utterances.map { |utt| utt['words'].length }
  
  puts "\n#{name} Split Analysis:"
  puts "  Utterances: #{utterances.length}"
  puts "  Speakers: #{speakers.length}"
  puts "  Word count - Min: #{word_counts.min}, Max: #{word_counts.max}, Avg: #{(word_counts.sum.to_f / word_counts.length).round(1)}"
  puts "  Speaker range: #{speakers.min}-#{speakers.max}"
end

analyze_split(train_split, "Train")
analyze_split(dev_split, "Dev")
analyze_split(test_split, "Test")

puts "\n✓ Balanced dataset created successfully!"
puts "All splits use the same speakers with different utterances"
puts "This should eliminate domain mismatch issues"
