#Normalize each coefficient across frames: mean→0, variance→1.

# file: lib/asr/features/cmvn.rb
module ASR
  module Features
    class CMVN
      def self.apply(features, eps: 1e-12)
        return [] if features.nil? || features.empty?
        t = features.length
        d = features[0].length

        mean = Array.new(d, 0.0)
        features.each { |v| (0...d).each { |j| mean[j] += v[j] } }
        (0...d).each { |j| mean[j] /= t.to_f }

        var = Array.new(d, 0.0)
        features.each do |v|
          (0...d).each do |j|
            diff = v[j] - mean[j]
            var[j] += diff * diff
          end
        end
        (0...d).each { |j| var[j] = var[j] / t.to_f }

        features.map do |v|
          (0...d).map do |j|
            denom = Math.sqrt(var[j] + eps)
            (v[j] - mean[j]) / denom
          end
        end
      end
    end
  end
end

