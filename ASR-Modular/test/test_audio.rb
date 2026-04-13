# frozen_string_literal: true

require "minitest/autorun"
require_relative "../lib/asr"

class TestAudio < Minitest::Test
  def test_write_and_read_wav
    dir = "tmp/test_audio"
    ASR::Utils.ensure_dir(dir)
    path = File.join(dir, "a.wav")

    sr = 16_000
    x = (0...(sr * 0.1)).map { |i| Math.sin(2.0 * Math::PI * 440.0 * i / sr.to_f) * 0.2 }
    ASR::Utils.write_wav_pcm16le(path, x, sr: sr)

    wav = ASR::Audio.read(path, target_sr: sr)
    assert_equal sr, wav[:sr]
    assert_in_delta x.length, wav[:samples].length, 2
  end
end
