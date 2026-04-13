# frozen_string_literal: true
# Predict one file using trained Word-HMM+GMM model + unigram LM.
# Usage:
#   ruby bin/predict_word_hmm_gmm.rb data/models/word_hmm_gmm.json data/lm/unigram.json data/lexicon/lexicon.txt path/to/test.wav
#
require "json"

require_relative "../lib/asr/audio/wav_reader"
require_relative "../lib/asr/preprocess/normalize"
require_relative "../lib/asr/preprocess/pre_emphasis"
require_relative "../lib/asr/preprocess/framing"
require_relative "../lib/asr/preprocess/windowing"
require_relative "../lib/asr/features/mfcc"
require_relative "../lib/asr/features/cmvn"

require_relative "../lib/asr/model/word_hmm"
require_relative "../lib/asr/decoder/viterbi"
require_relative "../lib/asr/lm/unigram"
require_relative "../lib/asr/lexicon/lexicon"

model_path = ARGV[0]
lm_path = ARGV[1]
lex_path = ARGV[2]
wav_path = ARGV[3]
abort("Usage: ruby bin/predict_word_hmm_gmm.rb data/models/word_hmm_gmm.json data/lm/unigram.json data/lexicon/lexicon.txt path/to/test.wav") if wav_path.nil?

payload = JSON.parse(File.read(model_path))
lm = ASR::LM::Unigram.load(lm_path)
lex = ASR::Lexicon::Lexicon.load(lex_path)

# MFCC config from model
sr = payload["mfcc"]["sr"]
mfcc = ASR::Features::MFCC.new(
  sr: sr,
  n_fft: payload["mfcc"]["n_fft"],
  n_mels: payload["mfcc"]["n_mels"],
  num_ceps: payload["mfcc"]["num_ceps"]
)

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
end

frames = extract_features(wav_path, mfcc)

lambda = 0.2 # LM weight (small for isolated words)

best = { word: nil, score: -1.0/0.0, hmm_score: nil, lm_score: nil }

payload["models"].each do |word, hmm_h|
  hmm = ASR::Model::WordHMM.from_h(hmm_h)
  v = ASR::Decoder::Viterbi.decode_word_hmm(hmm, frames)
  hmm_score = v[:score]
  lm_score = lm.logp(word)
  total = hmm_score + lambda * lm_score

  if total > best[:score]
    best = { word: word, score: total, hmm_score: hmm_score, lm_score: lm_score }
  end
end

puts "WAV: #{wav_path}"
puts "Prediction: #{best[:word]}"
puts "Scores: total=#{best[:score].round(3)} hmm=#{best[:hmm_score].round(3)} lm=#{best[:lm_score].round(3)}"
