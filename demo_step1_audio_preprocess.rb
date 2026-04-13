#This script loads one WAV, normalizes, pre-emphasizes, frames, applies Hamming window, and prints sanity checks.

# file: bin/demo_step1_audio_preprocess.rb
# Usage: ruby bin/demo_step1_audio_preprocess.rb path/to/file.wav

require_relative "../lib/asr/audio/wav_reader"
require_relative "../lib/asr/preprocess/normalize"
require_relative "../lib/asr/preprocess/pre_emphasis"
require_relative "../lib/asr/preprocess/framing"
require_relative "../lib/asr/preprocess/windowing"



path = ARGV[0]
abort("Usage: ruby bin/demo_step1_audio_preprocess.rb path/to/file.wav") if path.nil?

wav = ASR::Audio::WavReader.read(path, target_sr: 16000, force_mono: true)
samples = wav[:data][:samples]
sr = wav[:data][:sr]

samples_n = ASR::Preprocess::Normalize.peak(samples)
samples_p = ASR::Preprocess::PreEmphasis.apply(samples_n, alpha: 0.97)

fr = ASR::Preprocess::Framing.frame(samples_p, sr: sr, frame_ms: 25.0, hop_ms: 10.0)
frames = fr[:data][:frames]
frame_len = fr[:data][:frame_len]

win = ASR::Preprocess::Windowing.hamming(frame_len)
frames_w = ASR::Preprocess::Windowing.apply(frames, win)

puts "Loaded: #{wav[:meta][:path]}"
puts "Duration: #{wav[:meta][:duration_s].round(3)} s, sr=#{sr} Hz"
puts "Samples: #{samples.length}"
puts "Frames: #{frames.length}, frame_len=#{frame_len}, hop=#{fr[:data][:hop_len]}"
puts "First frame stats (after window): min=#{frames_w[0].min.round(6)} max=#{frames_w[0].max.round(6)}"

