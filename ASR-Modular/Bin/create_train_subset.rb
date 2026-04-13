#!/usr/bin/env ruby
# frozen_string_literal: true

require "json"

manifest_file = ARGV[0] || "Data/manifests/train_clean_100.jsonl"
subset_size = ARGV[1]&.to_i || 1000
output_file = ARGV[2] || "Data/manifests/train_clean_100_subset_#{subset_size}.jsonl"

puts "Creating train subset..."
puts "Source: #{manifest_file}"
puts "Subset size: #{subset_size}"
puts "Output: #{output_file}"

# Read all utterances
utterances = []
File.readlines(manifest_file).each do |line|
  utterances << JSON.parse(line)
end

puts "Total utterances available: #{utterances.length}"

# Take subset (evenly distributed)
if utterances.length > subset_size
  step = utterances.length.to_f / subset_size
  subset_utterances = (0...subset_size).map { |i| utterances[(i * step).floor] }
else
  subset_utterances = utterances
end

puts "Selected #{subset_utterances.length} utterances"

# Write subset manifest
File.open(output_file, "w") do |f|
  subset_utterances.each { |utt| f.puts(JSON.generate(utt)) }
end

puts "Subset manifest created: #{output_file}"

# Show sample
puts "\nSample utterances:"
subset_utterances.take(3).each_with_index do |utt, i|
  puts "  #{i+1}. #{utt['id']}: #{utt['text'][0..80]}..."
end
