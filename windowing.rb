#Goal: Apply a window function to each frame to reduce spectral leakage.

# file: lib/asr/preprocess/windowing.rb
module ASR
  module Preprocess
    class Windowing
      def self.hamming(n)
        raise ArgumentError, "n must be >= 1" if n < 1
        return [1.0] if n == 1
        (0...n).map { |i| 0.54 - 0.46 * Math.cos(2.0 * Math::PI * i / (n - 1)) }
      end

      def self.apply(frames, window)
        frames.map do |frame|
          frame.each_with_index.map { |x, i| x * window[i] }
        end
      end
    end
  end
end

