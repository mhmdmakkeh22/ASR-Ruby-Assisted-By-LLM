#This script runs: WAV → preprocess (Part 1) → MFCC → CMVN, then prints sanity checks.

# file: bin/demo_step2_mfcc.rb
# Usage: ruby bin/demo_step2_mfcc.rb data/raw_wav/test.wav

require_relative "../lib/asr/audio/wav_reader"
require_relative "../lib/asr/preprocess/normalize"
require_relative "../lib/asr/preprocess/pre_emphasis"
require_relative "../lib/asr/preprocess/framing"
require_relative "../lib/asr/preprocess/windowing"

require_relative "../lib/asr/features/mfcc"
require_relative "../lib/asr/features/cmvn"

path = ARGV[0]
abort("Usage: ruby bin/demo_step2_mfcc.rb data/raw_wav/test.wav") if path.nil?

wav = ASR::Audio::WavReader.read(path, target_sr: 16000, force_mono: true)
samples = wav[:data][:samples]
sr = wav[:data][:sr]

samples_n = ASR::Preprocess::Normalize.peak(samples)
samples_p = ASR::Preprocess::PreEmphasis.apply(samples_n, alpha: 0.97)

fr = ASR::Preprocess::Framing.frame(samples_p, sr: sr, frame_ms: 25.0, hop_ms: 10.0)
frames = fr[:data][:frames]
win = ASR::Preprocess::Windowing.hamming(fr[:data][:frame_len])
frames_w = ASR::Preprocess::Windowing.apply(frames, win)

mfcc = ASR::Features::MFCC.new(sr: sr, n_fft: 512, n_mels: 26, num_ceps: 13)
out = mfcc.transform_frames(frames_w)
features = out[:data][:features]
features_n = ASR::Features::CMVN.apply(features)

puts "Loaded: #{wav[:meta][:path]}"
puts "Frames: #{frames_w.length}"
puts "MFCC dim: #{out[:meta][:dim]}"
puts "First MFCC vector (raw): #{features[0].map { |x| x.round(4) }.join(', ')}"
puts "First MFCC vector (cmvn): #{features_n[0].map { |x| x.round(4) }.join(', ')}"

# Sanity checks
# mean approx 0, var approx 1 for each dim (rough check on first 3 dims)
t = features_n.length
d = features_n[0].length
means = Array.new(d, 0.0)
features_n.each { |v| (0...d).each { |j| means[j] += v[j] } }
means.map! { |m| m / t.to_f }

vars = Array.new(d, 0.0)
features_n.each do |v|
  (0...d).each do |j|
    diff = v[j] - means[j]
    vars[j] += diff * diff
  end
end
vars.map! { |vv| vv / t.to_f }

puts "CMVN check (first 3 dims):"
(0...[3, d].min).each do |j|
  puts "  dim#{j}: mean=#{means[j].round(3)} var=#{vars[j].round(3)}"
end
