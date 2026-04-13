#!/usr/bin/env ruby
# frozen_string_literal: true

# Evaluation script for test set after training on dev set
require "optparse"
require_relative "../lib/asr"

# Default configuration
opts = {
  model: nil,           # Trained acoustic model
  features_dir: nil,    # Test features directory  
  lexicon: nil,         # Lexicon file
  lm: nil,              # Language model file
  output: nil,           # Output file for hypotheses
  beam_width: 16,       # Beam search width
  lm_weight: 2.0,      # LM weight
  word_score: -1.0,     # Word insertion penalty
  verbose: true
}

OptionParser.new do |o|
  o.banner = "Usage: ruby bin/evaluate_test.rb --model MODEL --features_dir TEST_FEATURES --lexicon LEXICON --lm LM"
  o.on("--model PATH", "Trained acoustic model (.marshal)") { |v| opts[:model] = v }
  o.on("--features_dir DIR", "Test features directory") { |v| opts[:features_dir] = v }
  o.on("--lexicon PATH", "Lexicon TSV file") { |v| opts[:lexicon] = v }
  o.on("--lm PATH", "Language model JSON file") { |v| opts[:lm] = v }
  o.on("--output PATH", "Output hypotheses file") { |v| opts[:output] = v }
  o.on("--beam_width N", Integer, "Beam search width") { |v| opts[:beam_width] = v }
  o.on("--lm_weight F", Float, "Language model weight") { |v| opts[:lm_weight] = v }
  o.on("--word_score F", Float, "Word insertion penalty") { |v| opts[:word_score] = v }
  o.on("--[no-]verbose", "Show progress") { |v| opts[:verbose] = v }
end.parse!

# Validate inputs
missing = []
missing << "--model" if opts[:model].nil?
missing << "--features_dir" if opts[:features_dir].nil?
missing << "--lexicon" if opts[:lexicon].nil?
missing << "--lm" if opts[:lm].nil?

unless missing.empty?
  puts "Error: Missing required arguments: #{missing.join(', ')}"
  puts ""
  puts "Example usage:"
  puts "  ruby bin/evaluate_test.rb \\"
  puts "    --model models/dev/acoustic_balanced.marshal \\"
  puts "    --features_dir Data/features/test \\"
  puts "    --lexicon models/dev/lexicon.tsv \\"
  puts "    --lm models/dev/lm_unigram.json \\"
  puts "    --output test_hypotheses.txt"
  exit 1
end

# Check if files exist
unless File.exist?(opts[:model])
  puts "Error: Model file not found: #{opts[:model]}"
  exit 1
end

unless File.exist?(opts[:features_dir])
  puts "Error: Features directory not found: #{opts[:features_dir]}"
  exit 1
end

unless File.exist?(opts[:lexicon])
  puts "Error: Lexicon file not found: #{opts[:lexicon]}"
  exit 1
end

unless File.exist?(opts[:lm])
  puts "Error: Language model file not found: #{opts[:lm]}"
  exit 1
end

puts "=" * 60
puts "TEST SET EVALUATION"
puts "=" * 60
puts "Model:        #{opts[:model]}"
puts "Test data:    #{opts[:features_dir]}"
puts "Lexicon:     #{opts[:lexicon]}"
puts "Language model: #{opts[:lm]}"
puts "Output:       #{opts[:output]}"
puts "Beam width:   #{opts[:beam_width]}"
puts "LM weight:    #{opts[:lm_weight]}"
puts "Word score:   #{opts[:word_score]}"
puts "=" * 60

# Load models and data
puts "Loading models..."
am = ASR::Utils.marshal_load(opts[:model])
lexicon = ASR::Lexicon.load_tsv(opts[:lexicon])
lm_data = ASR::Utils.json_load(opts[:lm])
lm = ASR::LM::Unigram.from_h(lm_data)

puts "Loading test features..."
test_files = Dir.glob(File.join(opts[:features_dir], "*.marshal")).sort
puts "Found #{test_files.length} test utterances"

if test_files.empty?
  puts "Error: No test feature files found"
  exit 1
end

# Load test utterances
test_utterances = test_files.map do |fp|
  h = ASR::Utils.marshal_load(fp)
  { utt_id: h[:utt_id], words: ASR::Utils.tokenize(h[:text]), feats: h[:feats] }
end

puts "Initializing decoder..."
decoder = ASR::Decoder::BeamSearch.new(
  am: am,
  lexicon: lexicon,
  lm: lm,
  beam_size: opts[:beam_width],
  beam_delta: 20.0,
  lm_weight: opts[:lm_weight],
  word_penalty: opts[:word_score]
)

# Decode test set
puts "Decoding test set..."
start_time = Time.now

hypotheses = []
test_utterances.each_with_index do |utt, idx|
  if opts[:verbose] && (idx % 50 == 0 || idx == test_utterances.length - 1)
    progress = (idx + 1) * 100.0 / test_utterances.length
    puts "  Progress: #{progress.round(1)}% (#{idx + 1}/#{test_utterances.length}) - #{utt[:utt_id]}"
  end
  
  hyp_words = decoder.decode(utt[:feats])
  hypotheses << { utt_id: utt[:utt_id], ref: utt[:words], hyp: hyp_words }
end

decode_time = Time.now - start_time
puts "Decoding completed in #{decode_time.round(2)}s"

# Save hypotheses
puts "Saving hypotheses to #{opts[:output]}..."
File.open(opts[:output], 'w') do |f|
  hypotheses.each do |h|
    f.puts "#{h[:utt_id]}\t#{h[:hyp].join(' ')}"
  end
end

# Calculate WER
puts "Calculating Word Error Rate..."
total_words = 0
total_substitutions = 0
total_deletions = 0
total_insertions = 0

hypotheses.each do |h|
  ref = h[:ref]
  hyp = h[:hyp]
  
  # Calculate edit distance
  sub, del, ins = ASR::Eval.edit_distance(ref, hyp)
  
  total_words += ref.length
  total_substitutions += sub
  total_deletions += del
  total_insertions += ins
end

wer = (total_substitutions + total_deletions + total_insertions).to_f / total_words * 100.0

puts "=" * 60
puts "EVALUATION RESULTS"
puts "=" * 60
puts "Test utterances:  #{test_utterances.length}"
puts "Total reference words: #{total_words}"
puts "Substitutions: #{total_substitutions}"
puts "Deletions: #{total_deletions}"
puts "Insertions: #{total_insertions}"
puts "Word Error Rate (WER): #{wer.round(2)}%"
puts "Decoding time: #{decode_time.round(2)}s"
puts "Average time per utterance: #{(decode_time / test_utterances.length).round(3)}s"
puts "=" * 60

puts "Hypotheses saved to: #{opts[:output]}"
puts ""
puts "To view detailed results:"
puts "  head -20 #{opts[:output]}"
puts ""
puts "To compare with reference:"
puts "  paste <(cut -f2 Data/manifests/test_clean.jsonl | grep -o '[^\"]*' | head -20) <(head -20 #{opts[:output]})"
