# frozen_string_literal: true
# Train 5-state Word-HMM with GMM emissions using forced segmentation (equal-time chunks).
# Robust version:
# - Skips invalid WAV files (not RIFF/WAVE) instead of crashing
# - Skips files that fail resampling/reading
#
# Usage:
#   ruby bin/train_word_hmm_gmm.rb data/splits/train_manifest.json data/models/word_hmm_gmm.json
#
require "json"
require "fileutils"

require_relative "../lib/asr/audio/wav_reader"
require_relative "../lib/asr/preprocess/normalize"
require_relative "../lib/asr/preprocess/pre_emphasis"
require_relative "../lib/asr/preprocess/framing"
require_relative "../lib/asr/preprocess/windowing"
require_relative "../lib/asr/features/mfcc"
require_relative "../lib/asr/features/cmvn"

require_relative "../lib/asr/dataset/manifest"
require_relative "../lib/asr/model/gmm"
require_relative "../lib/asr/model/word_hmm"

train_manifest = ARGV[0]
out_model = ARGV[1]
abort("Usage: ruby bin/train_word_hmm_gmm.rb data/splits/train_manifest.json data/models/word_hmm_gmm.json") if train_manifest.nil? || out_model.nil?

entries = ASR::Dataset::Manifest.from_json(train_manifest)

# MFCC config
SR = 16000
N_FFT = 512
N_MELS = 26
NUM_CEPS = 13
K_GMM = 2      # start small; later try 4
EM_ITERS = 10  # reduce if training is slow

mfcc = ASR::Features::MFCC.new(sr: SR, n_fft: N_FFT, n_mels: N_MELS, num_ceps: NUM_CEPS)

def extract_features(path, mfcc)
  wav = ASR::Audio::WavReader.read(path, target_sr: 16000, force_mono: true)
  samples = wav[:data][:samples]
  sr = wav[:data][:sr]

  s1 = ASR::Preprocess::Normalize.peak(samples)
  s2 = ASR::Preprocess::PreEmphasis.apply(s1, alpha: 0.97)

  fr = ASR::Preprocess::Framing.frame(s2, sr: sr, frame_ms: 25.0, hop_ms: 10.0)
  frames = fr[:data][:frames]
  win = ASR::Preprocess::Windowing.hamming(fr[:data][:frame_len])
  frames_w = ASR::Preprocess::Windowing.apply(frames, win)

  out = mfcc.transform_frames(frames_w)
  feats = out[:data][:features]
  ASR::Features::CMVN.apply(feats)
rescue ASR::Audio::WavFormatError => e
  warn "SKIP (bad wav): #{path}  (#{e.message})"
  nil
rescue StandardError => e
  warn "SKIP (read/feature error): #{path}  (#{e.class}: #{e.message})"
  nil
end

# collect frames per word per state (forced segmentation into 5 equal chunks)
by_word = entries.group_by(&:label)
models = {}

by_word.each do |word, list|
  puts "Training word=#{word} (#{list.length} files in manifest)"
  state_frames = Array.new(5) { [] } # frames for each of 5 states
  used_files = 0

  list.each do |e|
    feats = extract_features(e.path, mfcc)
    next if feats.nil? || feats.empty?

    used_files += 1
    t = feats.length

    # split indices into 5 chunks
    (0...5).each do |s|
      a = (s * t) / 5
      b = ((s + 1) * t) / 5
      chunk = feats[a...b]
      state_frames[s].concat(chunk) if chunk && !chunk.empty?
    end
  end

  if used_files == 0
    warn "WARN: No usable files for word=#{word}. Skipping this word."
    next
  end

  # train one GMM per state
  state_gmms = state_frames.map.with_index do |frames, s_idx|
    if frames.empty?
      raise "No frames for word=#{word} state=#{s_idx}. (Maybe too few/short files after skipping invalid ones)"
    end
    ASR::Model::GMM.train(frames, k: K_GMM, iters: EM_ITERS, seed: 42 + s_idx)
  end

  hmm = ASR::Model::WordHMM.new(word: word, state_gmms: state_gmms)
  models[word.upcase] = hmm.to_h

  puts "  OK: used_files=#{used_files}, frames_per_state=#{state_frames.map(&:length).inspect}"
end

payload = {
  type: "word_hmm_gmm",
  mfcc: { sr: SR, n_fft: N_FFT, n_mels: N_MELS, num_ceps: NUM_CEPS, cmvn: true },
  topology: { states: 5, a_stay: 0.6, a_move: 0.4 },
  gmm: { k: K_GMM, em_iters: EM_ITERS },
  models: models
}

FileUtils.mkdir_p(File.dirname(out_model))
File.write(out_model, JSON.pretty_generate(payload))

puts "Saved model to #{out_model}"
puts "Trained words: #{models.keys.length}"

