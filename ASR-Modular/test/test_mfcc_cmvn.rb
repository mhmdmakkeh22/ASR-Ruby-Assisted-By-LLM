# frozen_string_literal: true

require "minitest/autorun"
require_relative "../lib/asr"

class TestMFCCCMVN < Minitest::Test
  def test_mfcc_shape_and_cmvn
    sr = 16_000
    t = (0...(sr * 0.5)).map { |i| Math.sin(2.0 * Math::PI * 440.0 * i / sr.to_f) * 0.2 }
    t = ASR::Preprocess.pre_emphasis(t)
    cfg = ASR::Features::MFCCConfig.new(
      target_sr: sr, n_mfcc: 13, n_fft: 512, win_len_ms: 25.0, hop_ms: 10.0, n_mels: 26, fmin: 0.0, fmax: nil
    )
    mfcc = ASR::Features.mfcc(t, sr, cfg)
    assert mfcc.length > 0
    assert_equal 13, mfcc.first.length

    norm = ASR::Features.cmvn_utterance(mfcc)
    # Rough mean check
    d = norm.first.length
    mean = Array.new(d, 0.0)
    norm.each { |v| d.times { |j| mean[j] += v[j] } }
    d.times { |j| mean[j] /= norm.length.to_f }
    mean.each { |m| assert_in_delta 0.0, m, 0.2 }
  end
end
