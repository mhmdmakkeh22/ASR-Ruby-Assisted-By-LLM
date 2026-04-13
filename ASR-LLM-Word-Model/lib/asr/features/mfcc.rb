#This module takes windowed frames and outputs MFCC features. Defaults: n_fft=512, n_mels=26, num_ceps=13. Optionally include delta and delta-delta later (Part 2.5).

# file: lib/asr/features/mfcc.rb
require_relative "fft"
require_relative "power_spectrum"
require_relative "mel_filterbank"
require_relative "dct"
require_relative "../preprocess/framing"
require_relative "../preprocess/windowing"
require_relative "../preprocess/pre_emphasis"

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
        # basic mel-cepstral features (array of arrays)
        feats = frames_w.map do |frame|
          spec = FFT.rfft_real(frame, @n_fft)
          pows = PowerSpectrum.from_fft(spec, @n_fft)
          mel  = MelFilterbank.apply(pows, @filters, eps: eps)
          logm = mel.map { |v| ::Math.log(v) }  # natural log
          DCT.dct2(logm, @num_ceps)
        end

        {
          data: { features: feats, sr: @sr, n_fft: @n_fft, n_mels: @n_mels, num_ceps: @num_ceps },
          meta: { feature_type: "mfcc", num_frames: feats.length, dim: @num_ceps }
        }
      end

      # Compute deltas for a sequence of frames (array of arrays)
      def self.compute_deltas(feats, n = 2)
        t = feats.length
        return [] if t == 0
        d = feats.first.length
        denom = (1..n).map { |m| m * m }.inject(0, &:+) * 2
        deltas = Array.new(t) { Array.new(d, 0.0) }
        (0...t).each do |i|
          (0...d).each do |k|
            num = 0.0
            (1..n).each do |m|
              ip = [t - 1, i + m].min
              im = [0, i - m].max
              num += m * (feats[ip][k] - feats[im][k])
            end
            deltas[i][k] = num / denom
          end
        end
        deltas
      end

      # Per-utterance Cepstral Mean and Variance Normalization
      def self.apply_cmvn_utterance!(feats)
        return feats if feats.nil? || feats.empty?
        t = feats.length
        d = feats.first.length
        means = Array.new(d, 0.0)
        vars  = Array.new(d, 0.0)

        (0...d).each do |k|
          feats.each { |f| means[k] += f[k] }
          means[k] /= t.to_f
        end

        (0...d).each do |k|
          feats.each { |f| vars[k] += (f[k] - means[k]) ** 2 }
          vars[k] = ::Math.sqrt(vars[k] / [t - 1, 1].max)
          vars[k] = 1.0 if vars[k] <= 0
        end

        feats.each do |f|
          (0...d).each { |k| f[k] = (f[k] - means[k]) / vars[k] }
        end
        feats
      end

      # Simple energy-based VAD: drop frames with energy below threshold_db relative to max
      # Accepts options such as `method:` and `threshold_db:`. Currently only `:energy` supported.
      def self.apply_vad!(frames_w, method: :energy, threshold_db: -40.0, **_opts)
        case method
        when :energy
          energies = frames_w.map { |fr| fr.map { |s| s * s }.inject(0.0, &:+) }
          max_e = energies.max || 1e-12
          keep = energies.map do |e|
            db = 10.0 * ::Math.log10([e, 1e-12].max / max_e)
            db > threshold_db
          end
          frames_w.each_with_index.map { |f, i| f if keep[i] }.compact
        else
          # unknown method: return frames unchanged
          frames_w
        end
      end

      # High-level convenience method: run full extract pipeline from raw samples
      # supports options hash as third arg: { vad: {...}, cmvn: :utterance, deltas: true|Integer }
      def self.extract(samples, sr, opts = {})
        raise ArgumentError, "samples empty" if samples.nil? || samples.empty?

        y = ASR::Preprocess::PreEmphasis.apply(samples)
        frames_res = ASR::Preprocess::Framing.frame(y, sr: sr)
        frames = frames_res[:data][:frames]
        frame_len = frames_res[:data][:frame_len]

        window = ASR::Preprocess::Windowing.hamming(frame_len)
        frames_w = ASR::Preprocess::Windowing.apply(frames, window)

        # optional VAD (expects hash of options forwarded to apply_vad!)
        if opts[:vad]
          frames_w = apply_vad!(frames_w, **opts[:vad])
        end

        mfcc = MFCC.new(sr: sr)
        res = mfcc.transform_frames(frames_w)
        feats = res[:data][:features]

        # optional CMVN
        if opts[:cmvn] == :utterance
          apply_cmvn_utterance!(feats)
        end

        # optional deltas (true -> N=2 or provide integer N)
        if opts[:deltas]
          n = opts[:deltas] == true ? 2 : opts[:deltas].to_i
          del = compute_deltas(feats, n)
          ddel = compute_deltas(del, n)
          feats = feats.each_with_index.map { |f, i| f + del[i] + ddel[i] }
        end

        res[:data][:features] = feats
        res[:meta][:dim] = feats.first ? feats.first.length : 0
        res
      end
    end
  end
end
