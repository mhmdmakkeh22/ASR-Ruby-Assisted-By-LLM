# frozen_string_literal: true

require "optparse"
require_relative "../lib/asr"

opts = {
  manifest: nil,
  out: "models/lexicon.tsv"
}

OptionParser.new do |o|
  o.banner = "Usage: ruby bin/build_lexicon.rb --manifest manifest.jsonl --out models/lexicon.tsv"
  o.on("--manifest PATH", "JSONL manifest") { |v| opts[:manifest] = v }
  o.on("--out PATH", "Output lexicon TSV") { |v| opts[:out] = v }
end.parse!

raise "Missing --manifest" if opts[:manifest].nil?

rows = ASR::Utils.read_jsonl(opts[:manifest])
ASR::Utils.ensure_dir(File.dirname(opts[:out]))
ASR::Lexicon.build_from_manifest(rows, out_path: opts[:out])

puts "Lexicon written: #{opts[:out]}"
