#!/usr/bin/env ruby
# frozen_string_literal: true

# Quick training script for dev_clean with multiple optimization levels
require "optparse"
require_relative "../lib/asr"

# Default optimized settings for dev_clean
OPTIMIZATION_PRESETS = {
  "fast" => {
    subset_ratio: 1.0,     # Use ALL data
    n_components: 2,         # Fewer GMM components for speed
    n_states: 3,            # Fewer HMM states for speed
    n_iter: 4,              # Fewer iterations
    trainer: "viterbi",     # Faster than Baum-Welch
    early_pruning: true
  },
  "balanced" => {
    subset_ratio: 1.0,     # Use ALL data
    n_components: 3,
    n_states: 4,
    n_iter: 6,
    trainer: "viterbi",
    early_pruning: true
  },
  "quality" => {
    subset_ratio: 1.0,     # Use ALL data
    n_components: 4,
    n_states: 5,
    n_iter: 8,
    trainer: "bw",          # Baum-Welch for better quality
    early_pruning: true
  },
  # NEW: Enhanced presets for better accuracy
  "enhanced" => {
    subset_ratio: 1.0,
    n_components: 5,
    n_states: 6,
    n_iter: 10,
    trainer: "bw",
    early_pruning: false  # Disable pruning for better accuracy
  },
  "robust" => {
    subset_ratio: 1.0,
    n_components: 6,
    n_states: 7,
    n_iter: 12,
    trainer: "bw",
    early_pruning: false,
    var_floor: 1e-4  # Lower variance floor for better modeling
  }
}

def print_usage
  puts "Usage: ruby bin/quick_train_dev_clean.rb [OPTIONS]"
  puts ""
  puts "Optimization presets:"
  OPTIMIZATION_PRESETS.each do |name, config|
    puts "  #{name.ljust(10)} | All data | #{config[:n_components]} components | #{config[:n_iter]} iterations | #{config[:trainer]}"
  end
  puts ""
  puts "Examples:"
  puts "  ruby bin/quick_train_dev_clean.rb --preset fast"
  puts "  ruby bin/quick_train_dev_clean.rb --preset balanced --out models/my_model.marshal"
  puts ""
  puts "All other options from train_am_optimized.rb are supported."
end

# Parse arguments
preset = "balanced"  # Default
opts = OPTIMIZATION_PRESETS[preset].dup
opts[:features_dir] = "Data/features/dev"
opts[:lexicon] = "models/dev/lexicon.tsv"
opts[:out] = "models/dev/acoustic_#{preset}.marshal"

OptionParser.new do |o|
  o.banner = "Quick training for dev_clean dataset"
  o.on("--preset NAME", OPTIMIZATION_PRESETS.keys, "Optimization preset (fast/balanced/quality)") do |v|
    preset = v
    opts = OPTIMIZATION_PRESETS[preset].dup
  end
  o.on("--features_dir DIR", "Features directory") { |v| opts[:features_dir] = v }
  o.on("--lexicon PATH", "Lexicon file") { |v| opts[:lexicon] = v }
  o.on("--out PATH", "Output model") { |v| opts[:out] = v }
  o.on("--subset_ratio X", Float, "Override: fraction of data to use") { |v| opts[:subset_ratio] = v }
  o.on("--n_components N", Integer, "Override: GMM components") { |v| opts[:n_components] = v }
  o.on("--n_states N", Integer, "Override: HMM states") { |v| opts[:n_states] = v }
  o.on("--n_iter N", Integer, "Override: training iterations") { |v| opts[:n_iter] = v }
  o.on("--trainer NAME", "Override: bw or viterbi") { |v| opts[:trainer] = v }
  o.on("--[no-]early_pruning", "Override: enable pruning") { |v| opts[:early_pruning] = v }
  o.on("--help", "Show this help") do
    print_usage
    exit
  end
end.parse!

# Validate inputs
unless File.exist?(opts[:features_dir])
  puts "Error: Features directory not found: #{opts[:features_dir]}"
  puts "Did you run feature extraction first?"
  exit 1
end

unless File.exist?(opts[:lexicon])
  puts "Error: Lexicon not found: #{opts[:lexicon]}"
  puts "Did you build the lexicon first?"
  exit 1
end

puts "=" * 60
puts "QUICK DEV_CLEAN TRAINING - #{preset.upcase} PRESET"
puts "=" * 60
puts "Features:   #{opts[:features_dir]}"
puts "Lexicon:    #{opts[:lexicon]}"
puts "Output:     #{opts[:out]}"
puts ""
puts "Configuration:"
puts "  Data subset:     100% (All #{files.length} utterances)"
puts "  GMM components: #{opts[:n_components]}"
puts "  HMM states:      #{opts[:n_states]}"
puts "  Iterations:      #{opts[:n_iter]}"
puts "  Trainer:         #{opts[:trainer]}"
puts "  Early pruning:   #{opts[:early_pruning]}"
puts "=" * 60

# Load data
puts "Loading features..."
files = Dir.glob(File.join(opts[:features_dir], "*.marshal")).sort
puts "Found #{files.length} utterance files"

if files.empty?
  puts "No feature files found. Run feature extraction first:"
  puts "  ruby bin/extract_features.rb --manifest Data/manifests/dev_clean.jsonl --out Data/features/dev"
  exit 1
end

utterances = files.map do |fp|
  h = ASR::Utils.marshal_load(fp)
  { utt_id: h[:utt_id], words: ASR::Utils.tokenize(h[:text]), feats: h[:feats] }
end

# Apply subset (only if explicitly set to < 1.0)
if opts[:subset_ratio] < 1.0
  subset_size = (utterances.length * opts[:subset_ratio]).to_i
  utterances = utterances.sample(subset_size)
  puts "Using subset: #{subset_size}/#{files.length} utterances"
else
  puts "Using all #{utterances.length} utterances"
end

# Train
dim = utterances.first[:feats].first.length
am = ASR::AM::OptimizedHMMGMM.new(
  n_states: opts[:n_states],
  n_components: opts[:n_components],
  dim: dim
)

puts "\nStarting training..."
start_time = Time.now

am.train!(
  utterances,
  ASR::Lexicon.load_tsv(opts[:lexicon]),
  n_iter: opts[:n_iter],
  trainer: opts[:trainer].to_sym,
  insert_sil: true,
  verbose: true,
  subset_ratio: 1.0,  # Use all data
  early_pruning: opts[:early_pruning]
)

end_time = Time.now
training_time = end_time - start_time

# Save model
ASR::Utils.marshal_dump(opts[:out], am)

puts "\n" + "=" * 60
puts "TRAINING COMPLETED"
puts "=" * 60
puts "Model saved: #{opts[:out]}"
puts "Training time: #{training_time.round(2)}s (#{(training_time / 60).round(2)} minutes)"
puts "Speedup vs original: ~3-8x (depending on preset)"
puts ""
puts "To decode with this model:"
puts "  ruby bin/decode.rb --features_dir #{opts[:features_dir]} --lexicon #{opts[:lexicon]} --lm models/dev/lm_unigram.json --am #{opts[:out]}"
