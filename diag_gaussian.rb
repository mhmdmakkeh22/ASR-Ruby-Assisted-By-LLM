# frozen_string_literal: true

module ASR
  module Model
    class DiagGaussian
      attr_reader :mean, :var

      def initialize(mean, var)
        @mean = mean # Array[Float] length D
        @var  = var  # Array[Float] length D (diagonal variances)
      end

      def dim
        @mean.length
      end

      # log N(x; mu, diag(var))
      def logpdf(x, eps: 1e-8)
        d = dim
        raise "dim mismatch" unless x.length == d
        s = 0.0
        (0...d).each do |j|
          v = @var[j] + eps
          diff = x[j] - @mean[j]
          s += ::Math.log(2.0 * ::Math::PI * v) + (diff * diff) / v
        end
        -0.5 * s
      end

      def to_h
        { mean: @mean, var: @var }
      end

      def self.from_h(h)
        new(h["mean"], h["var"])
      end
    end
  end
end
