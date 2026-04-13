# frozen_string_literal: true

require_relative "utils"
require_relative "preprocess"

module ASR
  module Features
    MFCCConfig = Struct.new(
      :target_sr,
      :n_mfcc,
      :n_fft,
      :win_len_ms,
      :hop_ms,
      :n_mels,
      :fmin,
      :fmax,
      keyword_init: true
    )

    DEFAULT_MFCC = MFCCConfig.new(
      target_sr: 16_000,
      n_mfcc: 13,
      n_fft: 512,
      win_len_ms: 25.0,
      hop_ms: 10.0,
      n_mels: 26,
      fmin: 0.0,
      fmax: 8_000.0
    )

    HIGH_RES_MFCC = MFCCConfig.new(
      target_sr: 16_000,
      n_mfcc: 20,
      n_fft: 512,
      win_len_ms: 25.0,
      hop_ms: 10.0,
      n_mels: 40,
      fmin: 0.0,
      fmax: 8_000.0
    )

    FAST_MFCC = MFCCConfig.new(
      target_sr: 16_000,
      n_mfcc: 13,
      n_fft: 256,
      win_len_ms: 20.0,
      hop_ms: 10.0,
      n_mels: 20,
      fmin: 0.0,
      fmax: 8_000.0
    )

    def self.hz_to_mel(f_hz)
      # Common HTK-style mel conversion; also widely used in MFCC pipelines.
      2595.0 * Math.log10(1.0 + f_hz.to_f / 700.0)
    end

    def self.mel_to_hz(m_mel)
      700.0 * (10**(m_mel.to_f / 2595.0) - 1.0)
    end

    def self.fft_recursive(x)
      n = x.length
      return x if n == 1
      raise ArgumentError, "FFT length must be power of 2" unless (n & (n - 1)).zero?

      even = fft_recursive((0...n).step(2).map { |i| x[i] })
      odd  = fft_recursive((1...n).step(2).map { |i| x[i] })

      half = n / 2
      out = Array.new(n)
      half.times do |k|
        angle = -2.0 * Math::PI * k / n
        tw = Complex(Math.cos(angle), Math.sin(angle))
        t = tw * odd[k]
        out[k] = even[k] + t
        out[k + half] = even[k] - t
      end
      out
    end

    def self.power_spectrum(frame, n_fft)
      a = frame.map { |v| Complex(v.to_f, 0.0) }
      if a.length < n_fft
        a += Array.new(n_fft - a.length, Complex(0.0, 0.0))
      elsif a.length > n_fft
        a = a[0, n_fft]
      end
      spec = fft_recursive(a)
      half = n_fft / 2
      (0..half).map do |k|
        c = spec[k]
        (c.real * c.real + c.imag * c.imag) / n_fft.to_f
      end
    end

    def self.mel_filterbank(sr:, n_fft:, n_mels:, fmin: 0.0, fmax: nil)
      fmax ||= sr / 2.0
      half_bins = n_fft / 2

      mel_min = hz_to_mel(fmin)
      mel_max = hz_to_mel(fmax)

      mel_points = (0..(n_mels + 1)).map do |i|
        mel_min + (mel_max - mel_min) * i / (n_mels + 1).to_f
      end
      hz_points = mel_points.map { |m| mel_to_hz(m) }
      bins = hz_points.map { |f| ((n_fft + 1) * f / sr.to_f).floor }.map { |b| [[b, 0].max, half_bins].min }

      fb = Array.new(n_mels) { Array.new(half_bins + 1, 0.0) }

      n_mels.times do |m|
        left = bins[m]
        center = bins[m + 1]
        right = bins[m + 2]
        next if right <= left

        (left..center).each do |k|
          denom = (center - left).to_f
          fb[m][k] = denom <= 0 ? 0.0 : (k - left) / denom
        end
        (center..right).each do |k|
          denom = (right - center).to_f
          fb[m][k] = denom <= 0 ? 0.0 : (right - k) / denom
        end
      end

      fb
    end

    def self.dct_type_ii(x, n_out)
      m = x.length
      out = Array.new(n_out, 0.0)
      n_out.times do |n|
        s = 0.0
        m.times do |i|
          s += x[i] * Math.cos(Math::PI * n * (i + 0.5) / m.to_f)
        end
        out[n] = s
      end
      out
    end

    # Returns frames: Array<Array<Float>> of shape T x n_mfcc
    def self.mfcc(samples, sr, cfg = DEFAULT_MFCC)
      raise ArgumentError, "sr mismatch: got #{sr}, expected #{cfg.target_sr}" unless sr == cfg.target_sr

      frames = ASR::Preprocess.frame_signal(samples, sr: sr, win_len_ms: cfg.win_len_ms, hop_ms: cfg.hop_ms, pad: true)
      win = ASR::Preprocess.hamming(frames.first.length)
      frames = ASR::Preprocess.apply_window(frames, win)

      n_fft = cfg.n_fft
      if n_fft < frames.first.length
        n_fft = ASR::Utils.next_pow2(frames.first.length)
      end
      raise ArgumentError, "n_fft must be power of 2" unless (n_fft & (n_fft - 1)).zero?

      fb = mel_filterbank(sr: sr, n_fft: n_fft, n_mels: cfg.n_mels, fmin: cfg.fmin, fmax: cfg.fmax)

      feats = frames.map do |fr|
        pxx = power_spectrum(fr, n_fft)
        mel_energies = fb.map do |w|
          s = 0.0
          w.each_with_index { |ww, k| s += ww * pxx[k] }
          Math.log([s, ASR::Utils::EPS].max)
        end
        dct_type_ii(mel_energies, cfg.n_mfcc)
      end

      feats
    end

    def self.cmvn_utterance(frames)
      return frames if frames.empty?
      t = frames.length
      d = frames.first.length

      mean = Array.new(d, 0.0)
      frames.each { |v| d.times { |j| mean[j] += v[j] } }
      d.times { |j| mean[j] /= t.to_f }

      var = Array.new(d, 0.0)
      frames.each do |v|
        d.times do |j|
          diff = v[j] - mean[j]
          var[j] += diff * diff
        end
      end
      d.times { |j| var[j] = Math.sqrt(var[j] / [t - 1, 1].max) }
      d.times { |j| var[j] = 1.0 if var[j] < 1e-8 }

      frames.map do |v|
        d.times.map { |j| (v[j] - mean[j]) / var[j] }
      end
    end
  end
end
