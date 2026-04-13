#!/usr/bin/env ruby
# frozen_string_literal: true

# Train on dev set and evaluate on test set
require "optparse"
require_relative "../lib/asr"

# Default configuration
opts = {
  preset: "balanced",    # fast/balanced/quality
  features_dir: "Data/features/dev",
  test_features_dir: "Data/features/test",
  lexicon: "models/dev/lexicon.tsv",
  lm: "models/dev/lm_unigram.json",
  model_dir: "models/dev",
  beam_width: 16,
  lm_weight: 2.0,
  word_score: -1.0,
  verbose: true
}

OptionParser.new do |o|
  o.banner = "Usage: ruby bin/train_and_eval.rb [OPTIONS]"
  o.on("--preset NAME", %w[fast balanced quality], "Training preset (fast/balanced/quality)") { |v| opts[:preset] = v }
  o.on("--features_dir DIR", "Dev features directory") { |v| opts[:features_dir] = v }
  o.on("--test_features_dir DIR", "Test features directory") { |v| opts[:test_features_dir] = v }
  o.on("--lexicon PATH", "Lexicon file") { |v| opts[:lexicon] = v }
  o.on("--lm PATH", "Language model file") { |v| opts[:lm] = v }
  o.on("--model_dir DIR", "Model output directory") { |v| opts[:model_dir] = v }
  o.on("--beam_width N", Integer, "Beam search width") { |v| opts[:beam_width] = v }
  o.on("--lm_weight F", Float, "Language model weight") { |v| opts[:lm_weight] = v }
  o.on("--word_score F", Float, "Word insertion penalty") { |v| opts[:word_score] = v }
  o.on("--[no-]verbose", "Show progress") { |v| opts[:verbose] = v }
end.parse!

puts "=" * 70
puts "TRAIN + EVAL PIPELINE"
puts "=" * 70
puts "Preset:        #{opts[:preset]}"
puts "Dev data:      #{opts[:features_dir]}"
puts "Test data:     #{opts[:test_features_dir]}"
puts "Lexicon:       #{opts[:lexicon]}"
puts "Language model: #{opts[:lm]}"
puts "Model dir:      #{opts[:model_dir]}"
puts "Beam width:    #{opts[:beam_width]}"
puts "=" * 70

# Step 1: Train on dev set
puts "\n" + "=" * 70
puts "STEP 1: TRAINING ON DEV SET"
puts "=" * 70

model_path = File.join(opts[:model_dir], "acoustic_#{opts[:preset]}.marshal")

# Build training command
train_cmd = [
  "ruby", "bin/quick_train_dev_clean.rb",
  "--preset", opts[:preset],
  "--features_dir", opts[:features_dir],
  "--lexicon", opts[:lexicon],
  "--out", model_path
]

puts "Running: #{train_cmd.join(' ')}"
train_success = system(*train_cmd)

unless train_success
  puts "Error: Training failed!"
  exit 1
end

puts "Training completed successfully!"
puts "Model saved to: #{model_path}"

# Step 2: Evaluate on test set
puts "\n" + "=" * 70
puts "STEP 2: EVALUATION ON TEST SET"
puts "=" * 70

# Check if test features exist
unless Dir.exist?(opts[:test_features_dir])
  puts "Error: Test features directory not found: #{opts[:test_features_dir]}"
  puts "Did you extract test features?"
  puts "Run: ruby bin/extract_features.rb --manifest Data/manifests/test_clean.jsonl --out #{opts[:test_features_dir]}"
  exit 1
end

test_output = File.join(opts[:model_dir], "test_hypotheses_#{opts[:preset]}.txt")

# Build evaluation command
eval_cmd = [
  "ruby", "bin/evaluate_test.rb",
  "--model", model_path,
  "--features_dir", opts[:test_features_dir],
  "--lexicon", opts[:lexicon],
  "--lm", opts[:lm],
  "--output", test_output,
  "--beam_width", opts[:beam_width].to_s,
  "--lm_weight", opts[:lm_weight].to_s,
  "--word_score", opts[:word_score].to_s
]

puts "Running: #{eval_cmd.join(' ')}"
eval_success = system(*eval_cmd)

unless eval_success
  puts "Error: Evaluation failed!"
  exit 1
end

puts "\n" + "=" * 70
puts "PIPELINE COMPLETED SUCCESSFULLY!"
puts "=" * 70
puts "Training preset: #{opts[:preset]}"
puts "Model: #{model_path}"
puts "Test results: #{test_output}"
puts ""
puts "Quick commands to view results:"
puts "  cat #{test_output}"
puts "  head -20 #{test_output}"
puts ""
puts "To run with different settings:"
puts "  ruby bin/train_and_eval.rb --preset fast"
puts "  ruby bin/train_and_eval.rb --preset quality --beam_width 32"
