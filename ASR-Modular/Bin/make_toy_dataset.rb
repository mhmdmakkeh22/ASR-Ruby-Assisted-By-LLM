# frozen_string_literal: true

require "optparse"
require_relative "../lib/asr"

# Generates a tiny synthetic dataset:
# - Each grapheme-phone (e.g., G_R) becomes a sine tone at a unique frequency.
# - SIL becomes zeros.
#
# This is NOT real speech, but it exercises the full ASR pipeline end-to-end.

opts = {
  out: "data/toy",
  sr: 16_000,
  phone_dur: 0.14, # seconds
  gap_dur: 0.02,
  amp: 0.3
}

OptionParser.new do |o|
  o.banner = "Usage: ruby bin/make_toy_dataset.rb [options]"
  o.on("--out PATH", "Output dir (default: #{opts[:out]})") { |v| opts[:out] = v }
  o.on("--sr N", Integer, "Sample rate (default: #{opts[:sr]})") { |v| opts[:sr] = v }
  o.on("--phone_dur SEC", Float, "Tone duration per phone (default: #{opts[:phone_dur]})") { |v| opts[:phone_dur] = v }
  o.on("--gap_dur SEC", Float, "Silence gap between phones (default: #{opts[:gap_dur]})") { |v| opts[:gap_dur] = v }
  o.on("--amp A", Float, "Amplitude (default: #{opts[:amp]})") { |v| opts[:amp] = v }
end.parse!

out_dir = opts[:out]
wav_dir = File.join(out_dir, "wav")
ASR::Utils.ensure_dir(wav_dir)

# 5 utterances, small but with repetition for stability.
utts = [
  ["utt0001", "RED RED RED"],
  ["utt0002", "BLUE BLUE BLUE"],
  ["utt0003", "RED BLUE RED"],
  ["utt0004", "BLUE RED BLUE"],
  ["utt0005", "RED BLUE BLUE RED"]
]

# Map each phone token to a frequency in Hz (deterministic)
def phone_freq(phone)
  return 0.0 if phone == "SIL"
  # Stable hash -> frequency bucket
  h = phone.each_byte.reduce(0) { |s, b| (s * 131 + b) & 0x7fffffff }
  250.0 + (h % 1400) # 250..1650 Hz
end

def synth_phone(phone, sr:, dur:, amp:)
  n = (sr * dur).to_i
  f = phone_freq(phone)
  return Array.new(n, 0.0) if f <= 0.0

  # Simple raised-cosine fade to avoid clicks
  fade = [ (0.01 * sr).to_i, 1 ].max
  out = Array.new(n, 0.0)
  n.times do |i|
    env =
      if i < fade
        0.5 - 0.5 * Math.cos(Math::PI * i / fade.to_f)
      elsif i > n - fade
        j = n - i
        0.5 - 0.5 * Math.cos(Math::PI * j / fade.to_f)
      else
        1.0
      end
    out[i] = amp * env * Math.sin(2.0 * Math::PI * f * i / sr.to_f)
  end
  out
end

def synth_utterance(text, sr:, phone_dur:, gap_dur:, amp:)
  words = ASR::Utils.tokenize(text)
  lex = ASR::Lexicon::LexiconStore.new({}) # empty; will use grapheme fallback
  phones = ["SIL"]
  words.each do |w|
    phones.concat(ASR::Lexicon.g2p_grapheme(w))
    phones << "SIL"
  end

  gap = Array.new((sr * gap_dur).to_i, 0.0)
  x = []
  phones.each do |p|
    x.concat(synth_phone(p, sr: sr, dur: phone_dur, amp: amp))
    x.concat(gap)
  end

  # Small noise
  x.map { |v| v + (rand - 0.5) * 0.005 }
end

# Write audio + transcripts.tsv
tsv_lines = []
utts.each do |utt_id, text|
  samples = synth_utterance(text, sr: opts[:sr], phone_dur: opts[:phone_dur], gap_dur: opts[:gap_dur], amp: opts[:amp])
  wav_path = File.join(wav_dir, "#{utt_id}.wav")
  ASR::Utils.write_wav_pcm16le(wav_path, samples, sr: opts[:sr])
  tsv_lines << "#{utt_id}\t#{text}"
end

ASR::Utils.write_lines(File.join(out_dir, "transcripts.tsv"), tsv_lines)
puts "Toy dataset written:"
puts "  wav_dir: #{wav_dir}"
puts "  transcripts: #{File.join(out_dir, 'transcripts.tsv')}"
puts "Utterances: #{utts.length}"
