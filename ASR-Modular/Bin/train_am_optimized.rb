# frozen_string_literal: true

require "optparse"
require_relative "../lib/asr"

opts = {
  features_dir: nil,
  lexicon: nil,
  out: "models/acoustic_optimized.marshal",
  n_states: 5,
  n_components: 4,
  n_iter: 8,
  trainer: "bw",     # bw or viterbi
  insert_sil: true,
  subset_ratio: 1.0,  # Use 100% of data by default
  early_pruning: true,
  use_optimized: true
}

OptionParser.new do |o|
  o.banner = "Usage: ruby bin/train_am_optimized.rb --features_dir data/features/split --lexicon models/lexicon.tsv --out models/acoustic_optimized.marshal"
  o.on("--features_dir DIR", "Dir containing <utt_id>.marshal feature files") { |v| opts[:features_dir] = v }
  o.on("--lexicon PATH", "Lexicon TSV") { |v| opts[:lexicon] = v }
  o.on("--out PATH", "Output acoustic model (Marshal)") { |v| opts[:out] = v }
  o.on("--n_states N", Integer, "Emitting states per phone") { |v| opts[:n_states] = v }
  o.on("--n_components N", Integer, "GMM mixtures per state") { |v| opts[:n_components] = v }
  o.on("--n_iter N", Integer, "Training iterations") { |v| opts[:n_iter] = v }
  o.on("--trainer NAME", "bw or viterbi") { |v| opts[:trainer] = v }
  o.on("--[no-]insert_sil", "Insert SIL between words and at ends") { |v| opts[:insert_sil] = v }
  o.on("--subset_ratio X", Float, "Fraction of data to use (0.1-1.0)") { |v| opts[:subset_ratio] = v }
  o.on("--[no-]early_pruning", "Enable early pruning of unlikely paths") { |v| opts[:early_pruning] = v }
  o.on("--[no-]use_optimized", "Use optimized HMM-GMM implementation") { |v| opts[:use_optimized] = v }
end.parse!

raise "Missing --features_dir" if opts[:features_dir].nil?
raise "Missing --lexicon" if opts[:lexicon].nil?

# Validate subset ratio
opts[:subset_ratio] = [[opts[:subset_ratio], 0.1].max, 1.0].min

lex = ASR::Lexicon.load_tsv(opts[:lexicon])

files = Dir.glob(File.join(opts[:features_dir], "*.marshal")).sort
raise "No feature files in #{opts[:features_dir]}" if files.empty?

puts "Loading #{files.length} utterances..."
utterances = files.map do |fp|
  h = ASR::Utils.marshal_load(fp)
  { utt_id: h[:utt_id], words: ASR::Utils.tokenize(h[:text]), feats: h[:feats] }
end

dim = utterances.first[:feats].first.length

# Choose implementation
am_class = opts[:use_optimized] ? ASR::AM::OptimizedHMMGMM : ASR::AM::HMMGMM
am = am_class.new(n_states: opts[:n_states], n_components: opts[:n_components], dim: dim)

puts "Training #{opts[:use_optimized] ? 'Optimized' : 'Standard'} HMM-GMM:"
puts "  utts=#{utterances.length} (using #{(opts[:subset_ratio] * 100).to_i}%)"
puts "  dim=#{dim} states=#{opts[:n_states]} comps=#{opts[:n_components]}"
puts "  trainer=#{opts[:trainer]} insert_sil=#{opts[:insert_sil]}"
puts "  early_pruning=#{opts[:early_pruning]}"

start_time = Time.now

am.train!(
  utterances,
  lex,
  n_iter: opts[:n_iter],
  trainer: opts[:trainer].to_sym,
  insert_sil: opts[:insert_sil],
  verbose: true,
  subset_ratio: opts[:subset_ratio],
  early_pruning: opts[:early_pruning]
)

end_time = Time.now
training_time = end_time - start_time

ASR::Utils.marshal_dump(opts[:out], am)
puts "Saved AM: #{opts[:out]}"
puts "Training time: #{training_time.round(2)} seconds (#{(training_time / 60).round(2)} minutes)"
