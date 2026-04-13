#Goal: Boost high frequencies before spectral analysis. Standard formula: y[n] = x[n] - α x[n-1], α≈0.97.

# file: lib/asr/preprocess/pre_emphasis.rb
module ASR
  module Preprocess
    class PreEmphasis
      def self.apply(samples, alpha: 0.97)
        return [] if samples.nil? || samples.empty?
        y = Array.new(samples.length, 0.0)
        y[0] = samples[0]
        (1...samples.length).each do |n|
          y[n] = samples[n] - alpha * samples[n - 1]
        end
        y
      end
    end
  end
end

