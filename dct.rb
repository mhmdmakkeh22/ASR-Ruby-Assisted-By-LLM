#Compute cepstral coefficients from log-mel energies using DCT-II.

# file: lib/asr/features/dct.rb
module ASR
  module Features
    class DCT
      # DCT-II for vector x (length N), return first num_ceps coefficients
      def self.dct2(x, num_ceps)
        n = x.length
        out = Array.new(num_ceps, 0.0)
        (0...num_ceps).each do |k|
          s = 0.0
          (0...n).each do |i|
            s += x[i] * Math.cos(Math::PI * k * (2*i + 1) / (2.0 * n))
          end
          out[k] = s
        end
        out
      end
    end
  end
end

