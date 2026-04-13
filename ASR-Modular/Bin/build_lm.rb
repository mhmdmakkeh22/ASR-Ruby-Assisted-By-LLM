# frozen_string_literal: true

require "optparse"
require_relative "../lib/asr"

opts = {
  manifest: nil,
  out: "models/lm_unigram.json",
  add_one: 1.0
}

OptionParser.new do |o|
  o.banner = "Usage: ruby bin/build_lm.rb --manifest manifest.jsonl --out models/lm_unigram.json"
  o.on("--manifest PATH", "JSONL manifest") { |v| opts[:manifest] = v }
  o.on("--out PATH", "Output LM JSON") { |v| opts[:out] = v }
  o.on("--add_one A", Float, "Laplace smoothing alpha (default: #{opts[:add_one]})") { |v| opts[:add_one] = v }
end.parse!

raise "Missing --manifest" if opts[:manifest].nil?

rows = ASR::Utils.read_jsonl(opts[:manifest])
texts = rows.map { |r| r["text"] }
lm = ASR::LM.build_unigram(texts, add_one: opts[:add_one])
ASR::LM.save_json(opts[:out], lm)

puts "Unigram LM written: #{opts[:out]} (vocab=#{lm.vocab.length})"
