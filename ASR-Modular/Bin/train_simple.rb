#!/usr/bin/env ruby
# frozen_string_literal: true

require "optparse"
require_relative "../lib/asr"

opts = {
  features_dir: "Data/features/dev",
  lexicon: "models/dev/lexicon.tsv",
  out: "models/dev/acoustic_simple.marshal",
  n_components: 2,
  n_states: 3,
  n_iter: 4,
  trainer: "viterbi"
}

OptionParser.new do |o|
  o.banner = "Usage: ruby bin/train_simple.rb [OPTIONS]"
  o.on("--features_dir DIR", "Features directory") { |v| opts[:features_dir] = v }
  o.on("--lexicon PATH", "Lexicon file") { |v| opts[:lexicon] = v }
  o.on("--out PATH", "Output model file") { |v| opts[:out] = v }
  o.on("--n_components N", Integer, "GMM components per state") { |v| opts[:n_components] = v }
  o.on("--n_states N", Integer, "HMM states per phone") { |v| opts[:n_states] = v }
  o.on("--n_iter N", Integer, "Training iterations") { |v| opts[:n_iter] = v }
  o.on("--trainer TYPE", %w[viterbi bw], "Training algorithm") { |v| opts[:trainer] = v }
end.parse!

puts "Simple Training Configuration"
puts "=" * 50
puts "Features: #{opts[:features_dir]}"
puts "Lexicon: #{opts[:lexicon]}"
puts "Output: #{opts[:out]}"
puts "Components: #{opts[:n_components]}"
puts "States: #{opts[:n_states]}"
puts "Iterations: #{opts[:n_iter]}"
puts "Trainer: #{opts[:trainer]}"
puts "=" * 50

# Load data
puts "Loading training data..."
files = Dir.glob(File.join(opts[:features_dir], "*.marshal")).sort
puts "Found #{files.length} utterances"

utterances = files.map do |fp|
  h = ASR::Utils.marshal_load(fp)
  { 
    utt_id: h[:utt_id], 
    words: ASR::Utils.tokenize(h[:text]),  # Tokenize text to words
    feats: h[:feats] 
  }
end

# Get feature dimension
feature_dim = utterances.first[:feats].first.length
puts "Feature dimension: #{feature_dim}"

# Load lexicon
puts "Loading lexicon..."
lexicon = ASR::Lexicon.load_tsv(opts[:lexicon])

# Create simple acoustic model (non-optimized for debugging)
puts "Creating acoustic model..."
am = ASR::AM::HMMGMM.new(
  n_states: opts[:n_states],
  n_components: opts[:n_components],
  dim: feature_dim,
  var_floor: 1e-3
)

# Train with detailed output
puts "Starting training..."
start_time = Time.now

opts[:n_iter].times do |iter|
  iter_start = Time.now
  puts "Iteration #{iter + 1}/#{opts[:n_iter]}"
  
  # Train one iteration
  am.train!(utterances, lexicon, n_iter: 1, trainer: opts[:trainer].to_sym)
  
  iter_time = Time.now - iter_start
  puts "  Iteration time: #{iter_time.round(2)}s"
end

total_time = Time.now - start_time
puts "Training completed in #{total_time.round(2)}s"

# Save model
puts "Saving model to #{opts[:out]}..."
ASR::Utils.marshal_dump(opts[:out], am)

puts "Simple training completed!"
puts "Model saved: #{opts[:out]}"
