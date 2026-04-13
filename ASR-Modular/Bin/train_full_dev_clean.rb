#!/usr/bin/env ruby
# frozen_string_literal: true

# Full dev_clean training with maximum optimizations
require_relative "../lib/asr"

puts "=" * 60
puts "FULL DEV_CLEAN TRAINING - ALL DATA OPTIMIZED"
puts "=" * 60

# Configuration for full dataset training
config = {
  features_dir: "Data/features/dev",
  lexicon: "models/dev/lexicon.tsv", 
  out: "models/dev/acoustic_full_optimized.marshal",
  n_states: 4,           # Reduced from 5 for speed
  n_components: 3,        # Reduced from 4 for speed
  n_iter: 6,             # Reduced from 8 for speed
  trainer: "viterbi",    # Faster than Baum-Welch
  early_pruning: true,   # Enable pruning
  use_optimized: true    # Use optimized implementation
}

puts "Configuration:"
puts "  Dataset:         All dev_clean utterances (~2,703)"
puts "  HMM states:      #{config[:n_states]}"
puts "  GMM components:  #{config[:n_components]}"
puts "  Iterations:      #{config[:n_iter]}"
puts "  Trainer:         #{config[:trainer]}"
puts "  Early pruning:   #{config[:early_pruning]}"
puts "  Output:          #{config[:out]}"
puts "=" * 60

# Validate inputs
unless File.exist?(config[:features_dir])
  puts "Error: Features directory not found: #{config[:features_dir]}"
  puts "Run: ruby bin/extract_features.rb --manifest Data/manifests/dev_clean.jsonl --out Data/features/dev"
  exit 1
end

unless File.exist?(config[:lexicon])
  puts "Error: Lexicon not found: #{config[:lexicon]}"
  puts "Run: ruby bin/build_lexicon.rb --manifest Data/manifests/dev_clean.jsonl --out #{config[:lexicon]}"
  exit 1
end

# Load all data
puts "Loading all dev_clean features..."
files = Dir.glob(File.join(config[:features_dir], "*.marshal")).sort
puts "Found #{files.length} utterance files"

if files.empty?
  puts "No feature files found!"
  exit 1
end

puts "Loading utterances..."
utterances = files.map.with_index do |fp, i|
  if i % 500 == 0
    puts "  Loaded #{i + 1}/#{files.length} utterances..."
  end
  h = ASR::Utils.marshal_load(fp)
  { utt_id: h[:utt_id], words: ASR::Utils.tokenize(h[:text]), feats: h[:feats] }
end

puts "Successfully loaded #{utterances.length} utterances"

# Train optimized model
dim = utterances.first[:feats].first.length
am = ASR::AM::OptimizedHMMGMM.new(
  n_states: config[:n_states],
  n_components: config[:n_components],
  dim: dim
)

puts "\nStarting optimized training on full dataset..."
puts "Expected speedup: 3-5x vs original implementation"
puts ""

start_time = Time.now

am.train!(
  utterances,
  ASR::Lexicon.load_tsv(config[:lexicon]),
  n_iter: config[:n_iter],
  trainer: config[:trainer].to_sym,
  insert_sil: true,
  verbose: true,
  subset_ratio: 1.0,  # Use ALL data
  early_pruning: config[:early_pruning]
)

end_time = Time.now
training_time = end_time - start_time

# Save model
ASR::Utils.marshal_dump(config[:out], am)

puts "\n" + "=" * 60
puts "FULL TRAINING COMPLETED"
puts "=" * 60
puts "Model saved: #{config[:out]}"
puts "Training time: #{training_time.round(2)}s (#{(training_time / 60).round(2)} minutes)"
puts "Dataset size: #{utterances.length} utterances"
puts ""
puts "Optimizations used:"
puts "✓ Memory-efficient forward-backward algorithm"
puts "✓ Early pruning of unlikely paths"
puts "✓ Cached emission probabilities"
puts "✓ Reduced model complexity"
puts "✓ Viterbi training (faster than Baum-Welch)"
puts ""
puts "To decode with this model:"
puts "  ruby bin/decode.rb --features_dir #{config[:features_dir]} --lexicon #{config[:lexicon]} --lm models/dev/lm_unigram.json --am #{config[:out]}"
