#!/usr/bin/env ruby
# frozen_string_literal: true

require "json"

def analyze_manifest(manifest_file, name)
  puts "\n=== #{name} Analysis ==="
  
  utterances = []
  File.readlines(manifest_file).each do |line|
    utterances << JSON.parse(line)
  end
  
  puts "Total utterances: #{utterances.length}"
  
  # Speaker distribution
  speakers = {}
  utterances.each do |utt|
    speaker = utt['id'].split('-')[0]
    speakers[speaker] ||= 0
    speakers[speaker] += 1
  end
  
  puts "Speakers: #{speakers.keys.length}"
  puts "Speaker range: #{speakers.keys.min}-#{speakers.keys.max}"
  
  # Word count distribution
  word_counts = utterances.map { |utt| utt['words'].length }
  puts "Word count - Min: #{word_counts.min}, Max: #{word_counts.max}, Avg: #{(word_counts.sum.to_f / word_counts.length).round(1)}"
  
  # Sample utterances
  puts "Sample utterances:"
  utterances.take(3).each_with_index do |utt, i|
    puts "  #{i+1}. #{utt['id']}: #{utt['text'][0..60]}..."
  end
  
  return utterances, speakers
end

# Analyze existing datasets
dev_utterances, dev_speakers = analyze_manifest("Data/manifests/dev_clean.jsonl", "Dev Clean")
test_utterances, test_speakers = analyze_manifest("Data/manifests/test_clean.jsonl", "Test Clean")
train_utterances, train_speakers = analyze_manifest("Data/manifests/train_clean_100.jsonl", "Train Clean-100")

# Check speaker overlap
puts "\n=== Speaker Overlap Analysis ==="
dev_speakers_set = dev_speakers.keys.to_set
test_speakers_set = test_speakers.keys.to_set
train_speakers_set = train_speakers.keys.to_set

puts "Speakers in Dev only: #{(dev_speakers_set - train_speakers_set).length}"
puts "Speakers in Test only: #{(test_speakers_set - train_speakers_set).length}"
puts "Speakers in Train only: #{(train_speakers_set - dev_speakers_set - test_speakers_set).length}"
puts "Speakers in all three: #{(dev_speakers_set & test_speakers_set & train_speakers_set).length}"

# Find common speakers for balanced split
common_speakers = (dev_speakers_set & test_speakers_set & train_speakers_set).to_a
puts "Common speakers (first 10): #{common_speakers.take(10).join(', ')}"

if common_speakers.length >= 10
  puts "\n✓ Good! We have #{common_speakers.length} common speakers for balanced split"
else
  puts "\n⚠️  Only #{common_speakers.length} common speakers - may need different strategy"
end
