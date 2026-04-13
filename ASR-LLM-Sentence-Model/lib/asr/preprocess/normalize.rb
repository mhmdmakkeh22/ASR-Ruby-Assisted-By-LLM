#Goal: Normalize amplitude to avoid scale differences between recordings. Peak normalization is simple and educational.
# file: lib/asr/preprocess/normalize.rb
module ASR
  module Preprocess
    class EmptyAudioError < StandardError; end

    class Normalize
      def self.peak(samples, eps: 1e-12)
        raise EmptyAudioError, "Empty samples" if samples.nil? || samples.empty?

        max_abs = samples.map(&:abs).max
        return samples.dup if max_abs.nil? || max_abs < eps

        scale = 1.0 / max_abs
        samples.map { |x| x * scale }
      end
    end
  end
end
