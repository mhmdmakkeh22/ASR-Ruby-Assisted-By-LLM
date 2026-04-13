# frozen_string_literal: true

require_relative "../math/logsumexp"
require_relative "diag_gaussian"

module ASR
  module Model
    class GMM
      attr_reader :weights, :gaussians

      def initialize(weights, gaussians)
        @weights = weights # Array[Float] sum to 1, length K
        @gaussians = gaussians # Array[DiagGaussian] length K
        @logw = @weights.map { |w| ::Math.log([w, 1e-12].max) }
      end

      def k
        @gaussians.length
      end

      def dim
        @gaussians[0].dim
      end

      # log p(x) = log sum_k w_k * N_k(x)
      def logpdf(x)
        terms = []
        @gaussians.each_with_index do |g, i|
          terms << (@logw[i] + g.logpdf(x))
        end
        ASR::MathUtil.logsumexp(terms)
      end

      def loglik(frames)
        frames.sum { |x| logpdf(x) }
      end

      # ----- Training (EM) -----
      def self.train(frames, k:, iters: 15, seed: 0)
        raise "Need frames" if frames.nil? || frames.empty?
        srand(seed)

        d = frames[0].length
        # init means by sampling frames
        means = frames.sample(k).map(&:dup)
        # init variances as global variance
        global_mean = Array.new(d, 0.0)
        frames.each { |x| (0...d).each { |j| global_mean[j] += x[j] } }
        (0...d).each { |j| global_mean[j] /= frames.length.to_f }

        global_var = Array.new(d, 0.0)
        frames.each do |x|
          (0...d).each do |j|
            diff = x[j] - global_mean[j]
            global_var[j] += diff * diff
          end
        end
        (0...d).each { |j| global_var[j] = global_var[j] / frames.length.to_f + 1e-3 }

        vars = Array.new(k) { global_var.dup }
        weights = Array.new(k, 1.0 / k)

        gmm = new(weights, means.map.with_index { |m, i| DiagGaussian.new(m, vars[i]) })

        iters.times do
          # E-step: responsibilities r[n][k] in log domain then normalize
          nk = Array.new(k, 0.0)
          sum_x = Array.new(k) { Array.new(d, 0.0) }
          sum_x2 = Array.new(k) { Array.new(d, 0.0) }

          frames.each do |x|
            logp = (0...k).map { |i| ::Math.log([gmm.weights[i], 1e-12].max) + gmm.gaussians[i].logpdf(x) }
            logden = ASR::MathUtil.logsumexp(logp)

            (0...k).each do |i|
              r_ni = ::Math.exp(logp[i] - logden) # responsibility
              nk[i] += r_ni
              (0...d).each do |j|
                sum_x[i][j]  += r_ni * x[j]
                sum_x2[i][j] += r_ni * x[j] * x[j]
              end
            end
          end

          n_total = frames.length.to_f
          weights = nk.map { |v| [v / n_total, 1e-12].max }
          # renormalize weights
          ws = weights.sum
          weights.map! { |w| w / ws }

          gaussians = []
          (0...k).each do |i|
            denom = [nk[i], 1e-12].max
            mean = (0...d).map { |j| sum_x[i][j] / denom }
            var  = (0...d).map do |j|
              ex2 = sum_x2[i][j] / denom
              v = ex2 - mean[j] * mean[j]
              [v, 1e-3].max
            end
            gaussians << DiagGaussian.new(mean, var)
          end

          gmm = new(weights, gaussians)
        end

        gmm
      end

      def to_h
        {
          weights: @weights,
          gaussians: @gaussians.map(&:to_h)
        }
      end

      def self.from_h(h)
        weights = h["weights"]
        gaussians = h["gaussians"].map { |gh| DiagGaussian.from_h(gh) }
        new(weights, gaussians)
      end
    end
  end
end
