#We implement a simple radix-2 FFT using Ruby’s built-in Complex. This is educational and sufficient for small-vocabulary experiments, though not the fastest.

# file: lib/asr/features/fft.rb
require "complex"

module ASR
  module Features
    class FFT
      def self.next_pow2(n)
        p = 1
        p <<= 1 while p < n
        p
      end

      # Recursive Cooley–Tukey FFT (radix-2). Input: array of Complex.
      def self.fft(x)
        n = x.length
        raise ArgumentError, "FFT length must be power of 2" unless (n & (n - 1)) == 0
        return x if n == 1

        even = []
        odd  = []
        x.each_with_index do |v, i|
          (i.even? ? even : odd) << v
        end

        fe = fft(even)
        fo = fft(odd)

        out = Array.new(n)
        (0...(n/2)).each do |k|
          tw = Complex.polar(1.0, -2.0 * ::Math::PI * k / n)
          t  = tw * fo[k]
          out[k]       = fe[k] + t
          out[k + n/2] = fe[k] - t
        end
        out
      end

      # Real-input FFT helper: returns complex spectrum for zero-padded frame
      def self.rfft_real(frame, n_fft)
        raise ArgumentError, "n_fft must be power of 2" unless (n_fft & (n_fft - 1)) == 0
        x = Array.new(n_fft, Complex(0.0, 0.0))
        frame_len = [frame.length, n_fft].min
        (0...frame_len).each { |i| x[i] = Complex(frame[i], 0.0) }
        fft(x)
      end
    end
  end
end
