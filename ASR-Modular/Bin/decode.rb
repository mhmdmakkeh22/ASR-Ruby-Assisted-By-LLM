# frozen_string_literal: true

require "optparse"
require_relative "../lib/asr"

opts = {
  features_dir: nil,
  lexicon: nil,
  lm: nil,
  am: nil,
  out: "exp/hyp.txt",
  beam_size: 200,
  beam_delta: 20.0,
  lm_weight: 0.8,
  word_penalty: -0.2
}

OptionParser.new do |o|
  o.banner = "Usage: ruby bin/decode.rb --features_dir data/features/split --lexicon models/lexicon.tsv --lm models/lm_unigram.json --am models/acoustic.marshal --out exp/hyp.txt"
  o.on("--features_dir DIR", "Dir containing <utt_id>.marshal") { |v| opts[:features_dir] = v }
  o.on("--lexicon PATH", "Lexicon TSV") { |v| opts[:lexicon] = v }
  o.on("--lm PATH", "Unigram LM JSON") { |v| opts[:lm] = v }
  o.on("--am PATH", "Acoustic model Marshal") { |v| opts[:am] = v }
  o.on("--out PATH", "Output hypotheses file") { |v| opts[:out] = v }
  o.on("--beam_size N", Integer, "Beam size") { |v| opts[:beam_size] = v }
  o.on("--beam_delta X", Float, "Keep hyps within best - delta") { |v| opts[:beam_delta] = v }
  o.on("--lm_weight X", Float, "LM weight") { |v| opts[:lm_weight] = v }
  o.on("--word_penalty X", Float, "Word insertion penalty") { |v| opts[:word_penalty] = v }
end.parse!

%i[features_dir lexicon lm am].each do |k|
  raise "Missing --#{k}" if opts[k].nil?
end

lex = ASR::Lexicon.load_tsv(opts[:lexicon])
lm = ASR::LM.load_json(opts[:lm])
am = ASR::Utils.marshal_load(opts[:am])

decoder = ASR::Decoder::BeamSearch.new(
  am: am,
  lexicon: lex,
  lm: lm,
  beam_size: opts[:beam_size],
  beam_delta: opts[:beam_delta],
  lm_weight: opts[:lm_weight],
  word_penalty: opts[:word_penalty]
)

files = Dir.glob(File.join(opts[:features_dir], "*.marshal")).sort
raise "No feature files in #{opts[:features_dir]}" if files.empty?

ASR::Utils.ensure_dir(File.dirname(opts[:out]))
lines = []

files.each_with_index do |fp, idx|
  h = ASR::Utils.marshal_load(fp)
  utt = h[:utt_id]
  hyp_words = decoder.decode(h[:feats])
  lines << "#{utt}\t#{hyp_words.join(' ')}"
  puts format("[%d/%d] %s => %s", idx + 1, files.length, utt, hyp_words.join(" "))
end

ASR::Utils.write_lines(opts[:out], lines)
puts "Wrote hypotheses: #{opts[:out]}"
