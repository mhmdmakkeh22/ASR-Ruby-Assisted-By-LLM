#Compute the power spectrum from FFT bins. We typically keep bins 0..n_fft/2 (inclusive).

# file: lib/asr/features/power_spectrum.rb
module ASR
  module Features
    class PowerSpectrum
      def self.from_fft(spec, n_fft)
        half = n_fft / 2
        # power for bins 0..half
        (0..half).map do |k|
          re = spec[k].real
          im = spec[k].imag
          (re * re + im * im) / n_fft.to_f
        end
      end
    end
  end
end

