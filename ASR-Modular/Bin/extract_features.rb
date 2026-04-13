# frozen_string_literal: true

require "optparse"
require_relative "../lib/asr"

opts = {
  manifest: nil,
  out_dir: "data/features/train",
  target_sr: 16_000,
  n_mfcc: 13,
  n_fft: 512,
  win_len_ms: 25.0,
  hop_ms: 10.0,
  n_mels: 26,
  preemph: 0.97
}

OptionParser.new do |o|
  o.banner = "Usage: ruby bin/extract_features.rb --manifest manifest.jsonl --out_dir data/features/split"
  o.on("--manifest PATH", "JSONL manifest") { |v| opts[:manifest] = v }
  o.on("--out_dir DIR", "Output features dir") { |v| opts[:out_dir] = v }
  o.on("--target_sr N", Integer, "Target sample rate") { |v| opts[:target_sr] = v }
  o.on("--n_mfcc N", Integer, "MFCC dim") { |v| opts[:n_mfcc] = v }
  o.on("--n_fft N", Integer, "FFT size") { |v| opts[:n_fft] = v }
  o.on("--win_len_ms X", Float, "Window length (ms)") { |v| opts[:win_len_ms] = v }
  o.on("--hop_ms X", Float, "Hop length (ms)") { |v| opts[:hop_ms] = v }
  o.on("--n_mels N", Integer, "Mel bands") { |v| opts[:n_mels] = v }
  o.on("--preemph A", Float, "Pre-emphasis coef") { |v| opts[:preemph] = v }
end.parse!

raise "Missing --manifest" if opts[:manifest].nil?

rows = ASR::Utils.read_jsonl(opts[:manifest])
ASR::Utils.ensure_dir(opts[:out_dir])

cfg = ASR::Features::MFCCConfig.new(
  target_sr: opts[:target_sr],
  n_mfcc: opts[:n_mfcc],
  n_fft: opts[:n_fft],
  win_len_ms: opts[:win_len_ms],
  hop_ms: opts[:hop_ms],
  n_mels: opts[:n_mels],
  fmin: 0.0,
  fmax: nil
)

rows.each_with_index do |r, idx|
  utt_id = r["utt_id"] || r["id"]
  raise "Missing utt_id/id in manifest row: #{r.inspect}" if utt_id.nil? || utt_id.empty? 
  path = r["audio_path"]
  text = r["text"]

  audio = ASR::Audio.read(path, target_sr: cfg.target_sr, force_mono: true, allow_ffmpeg_resample: true)
  x = ASR::Preprocess.normalize_peak(audio[:samples])
  x = ASR::Preprocess.pre_emphasis(x, coef: opts[:preemph])

  mfcc = ASR::Features.mfcc(x, audio[:sr], cfg)
  mfcc = ASR::Features.cmvn_utterance(mfcc)

  out_path = File.join(opts[:out_dir], "#{utt_id}.marshal")
  ASR::Utils.marshal_dump(out_path, { utt_id: utt_id, text: text, feats: mfcc })

  puts format("[%d/%d] feats OK utt=%s T=%d D=%d", idx + 1, rows.length, utt_id, mfcc.length, mfcc.first.length)
end

puts "Features written to: #{opts[:out_dir]}"
