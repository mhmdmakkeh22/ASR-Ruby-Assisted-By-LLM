# frozen_string_literal: true

require "optparse"
require_relative "../lib/asr"

opts = {
  audio_dir: nil,
  transcripts: nil,
  out: "data/manifests/train.jsonl"
}

OptionParser.new do |o|
  o.banner = "Usage: ruby bin/make_manifest.rb --audio_dir DIR --transcripts transcripts.tsv --out manifest.jsonl"
  o.on("--audio_dir DIR", "Directory with audio files named <utt_id>.wav or .flac") { |v| opts[:audio_dir] = v }
  o.on("--transcripts PATH", "TSV: utt_id<TAB>text") { |v| opts[:transcripts] = v }
  o.on("--out PATH", "Output JSONL manifest") { |v| opts[:out] = v }
end.parse!

raise "Missing --audio_dir" if opts[:audio_dir].nil?
raise "Missing --transcripts" if opts[:transcripts].nil?

rows = []
ASR::Utils.read_tsv(opts[:transcripts]).each do |utt_id, text|
  next if utt_id.nil? || text.nil?
  utt_id = utt_id.strip
  next if utt_id.empty?

  wav = File.join(opts[:audio_dir], "#{utt_id}.wav")
  flac = File.join(opts[:audio_dir], "#{utt_id}.flac")
  audio_path =
    if File.exist?(wav) then wav
    elsif File.exist?(flac) then flac
    else
      warn "SKIP missing audio for utt_id=#{utt_id}"
      next
    end

  rows << {
    "utt_id" => utt_id,
    "audio_path" => audio_path,
    "text" => text
  }
end

ASR::Utils.write_jsonl(opts[:out], rows)
puts "Wrote manifest: #{opts[:out]} (#{rows.length} utterances)"
