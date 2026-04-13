# frozen_string_literal: true

require "optparse"
require_relative "../lib/asr"

opts = {
  manifest: nil,
  hyp: nil,
  show: false
}

OptionParser.new do |o|
  o.banner = "Usage: ruby bin/eval_wer.rb --manifest manifest.jsonl --hyp exp/hyp.txt"
  o.on("--manifest PATH", "JSONL manifest") { |v| opts[:manifest] = v }
  o.on("--hyp PATH", "Hypotheses file: utt_id<TAB>decoded text") { |v| opts[:hyp] = v }
  o.on("--show", "Print per-utterance details") { opts[:show] = true }
end.parse!

raise "Missing --manifest" if opts[:manifest].nil?
raise "Missing --hyp" if opts[:hyp].nil?

refs = {}
ASR::Utils.read_jsonl(opts[:manifest]).each do |r|
  refs[r["utt_id"]] = ASR::Utils.tokenize(r["text"])
end

hyps = {}
ASR::Utils.read_tsv(opts[:hyp]).each do |utt_id, hyp_text|
  hyps[utt_id] = ASR::Utils.tokenize(hyp_text.to_s)
end

sum_sub = sum_ins = sum_del = sum_ref = 0

refs.each do |utt, ref_words|
  hyp_words = hyps.fetch(utt, [])
  res = ASR::Eval.wer(ref_words, hyp_words)
  sum_sub += res.sub
  sum_ins += res.ins
  sum_del += res.del
  sum_ref += res.ref_len

  if opts[:show]
    puts "#{utt}"
    puts "  REF: #{ref_words.join(' ')}"
    puts "  HYP: #{hyp_words.join(' ')}"
    puts format("  S=%d I=%d D=%d  WER=%.3f", res.sub, res.ins, res.del, res.wer)
  end
end

wer = (sum_sub + sum_ins + sum_del) / [sum_ref, 1].max.to_f
puts format("WER=%.3f  (S=%d I=%d D=%d)  N=%d", wer, sum_sub, sum_ins, sum_del, sum_ref)
