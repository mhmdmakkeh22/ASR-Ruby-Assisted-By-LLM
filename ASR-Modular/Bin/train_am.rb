# frozen_string_literal: true

require "optparse"
require_relative "../lib/asr"

opts = {
  features_dir: nil,
  lexicon: nil,
  out: "models/acoustic.marshal",
  n_states: 5,
  n_components: 4,
  n_iter: 8,
  trainer: "bw",     # bw or viterbi
  insert_sil: true
}

OptionParser.new do |o|
  o.banner = "Usage: ruby bin/train_am.rb --features_dir data/features/split --lexicon models/lexicon.tsv --out models/acoustic.marshal"
  o.on("--features_dir DIR", "Dir containing <utt_id>.marshal feature files") { |v| opts[:features_dir] = v }
  o.on("--lexicon PATH", "Lexicon TSV") { |v| opts[:lexicon] = v }
  o.on("--out PATH", "Output acoustic model (Marshal)") { |v| opts[:out] = v }
  o.on("--n_states N", Integer, "Emitting states per phone") { |v| opts[:n_states] = v }
  o.on("--n_components N", Integer, "GMM mixtures per state") { |v| opts[:n_components] = v }
  o.on("--n_iter N", Integer, "Training iterations") { |v| opts[:n_iter] = v }
  o.on("--trainer NAME", "bw or viterbi") { |v| opts[:trainer] = v }
  o.on("--[no-]insert_sil", "Insert SIL between words and at ends") { |v| opts[:insert_sil] = v }
end.parse!

raise "Missing --features_dir" if opts[:features_dir].nil?
raise "Missing --lexicon" if opts[:lexicon].nil?

lex = ASR::Lexicon.load_tsv(opts[:lexicon])

files = Dir.glob(File.join(opts[:features_dir], "*.marshal")).sort
raise "No feature files in #{opts[:features_dir]}" if files.empty?

utterances = files.map do |fp|
  h = ASR::Utils.marshal_load(fp)
  { utt_id: h[:utt_id], words: ASR::Utils.tokenize(h[:text]), feats: h[:feats] }
end

dim = utterances.first[:feats].first.length
am = ASR::AM::HMMGMM.new(n_states: opts[:n_states], n_components: opts[:n_components], dim: dim)

puts "Training HMM-GMM:"
puts "  utts=#{utterances.length} dim=#{dim} states=#{opts[:n_states]} comps=#{opts[:n_components]} trainer=#{opts[:trainer]} insert_sil=#{opts[:insert_sil]}"

am.train!(
  utterances,
  lex,
  n_iter: opts[:n_iter],
  trainer: opts[:trainer].to_sym,
  insert_sil: opts[:insert_sil],
  verbose: true
)

ASR::Utils.marshal_dump(opts[:out], am)
puts "Saved AM: #{opts[:out]}"
