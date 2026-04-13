#This module takes windowed frames and outputs MFCC features. Defaults: n_fft=512, n_mels=26, num_ceps=13. Optionally include delta and delta-delta later (Part 2.5).

# file: lib/asr/features/mfcc.rb
require_relative "fft"
require_relative "power_spectrum"
require_relative "mel_filterbank"
require_relative "dct"

module ASR
  module Features
    class MFCC
      def initialize(sr:, n_fft: 512, n_mels: 26, num_ceps: 13, fmin: 0.0, fmax: nil)
        @sr = sr
        @n_fft = n_fft
        @n_mels = n_mels
        @num_ceps = num_ceps
        @fmin = fmin
        @fmax = fmax || sr / 2.0
        @filters = MelFilterbank.build(sr: sr, n_fft: n_fft, n_mels: n_mels, fmin: fmin, fmax: @fmax)
      end

      def transform_frames(frames_w, eps: 1e-12)
        feats = frames_w.map do |frame|
          spec = FFT.rfft_real(frame, @n_fft)
          pows = PowerSpectrum.from_fft(spec, @n_fft)
          mel  = MelFilterbank.apply(pows, @filters, eps: eps)
          logm = mel.map { |v| Math.log(v) }  # natural log
          DCT.dct2(logm, @num_ceps)
        end

        {
          data: { features: feats, sr: @sr, n_fft: @n_fft, n_mels: @n_mels, num_ceps: @num_ceps },
          meta: { feature_type: "mfcc", num_frames: feats.length, dim: @num_ceps }
        }
      end
    end
  end
end
