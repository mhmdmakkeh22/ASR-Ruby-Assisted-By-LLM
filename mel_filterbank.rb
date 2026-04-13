#We build triangular mel filters between fmin and fmax. Standard settings: n_mels=26, fmin=0, fmax=sr/2. Output: mel energies per frame.

# file: lib/asr/features/mel_filterbank.rb
module ASR
  module Features
    class MelFilterbank
      def self.hz_to_mel(hz)
        2595.0 * Math.log10(1.0 + hz / 700.0)
      end

      def self.mel_to_hz(mel)
        700.0 * (10.0**(mel / 2595.0) - 1.0)
      end

      # Returns filterbank as array of filters, each filter is array length (n_fft/2+1)
      def self.build(sr:, n_fft:, n_mels: 26, fmin: 0.0, fmax: nil)
        fmax ||= sr / 2.0
        n_bins = n_fft / 2 + 1

        mel_min = hz_to_mel(fmin)
        mel_max = hz_to_mel(fmax)

        # n_mels filters need n_mels + 2 points
        mel_points = (0...(n_mels + 2)).map do |i|
          mel_min + (mel_max - mel_min) * i / (n_mels + 1).to_f
        end

        hz_points = mel_points.map { |m| mel_to_hz(m) }
        bin_points = hz_points.map { |hz| ((n_fft + 1) * hz / sr).floor }

        filters = []
        (1..n_mels).each do |m|
          f = Array.new(n_bins, 0.0)
          left  = bin_points[m - 1]
          center= bin_points[m]
          right = bin_points[m + 1]

          # safety clamp
          left = [[left, 0].max, n_bins-1].min
          center = [[center, 0].max, n_bins-1].min
          right = [[right, 0].max, n_bins-1].min

          # rising slope
          (left...center).each do |k|
            denom = (center - left).to_f
            f[k] = denom > 0 ? (k - left) / denom : 0.0
          end
          # falling slope
          (center...right).each do |k|
            denom = (right - center).to_f
            f[k] = denom > 0 ? (right - k) / denom : 0.0
          end
          filters << f
        end
        filters
      end

      def self.apply(power_spec, filters, eps: 1e-12)
        filters.map do |f|
          s = 0.0
          power_spec.each_with_index { |p, k| s += p * f[k] }
          s < eps ? eps : s
        end
      end
    end
  end
end

