# file: lib/asr/math/gmm.rb
require "numo/narray"
require_relative "logsumexp"

module ASR
  module Math
    class GMM
      attr_reader :n_components, :dim, :weights, :means, :covs

      def initialize(n_components, dim)
        @n_components = n_components
        @dim = dim
        @weights = Numo::DFloat.zeros(n_components).fill(1.0 / n_components)
        @means   = Numo::DFloat.zeros(n_components, dim)
        @covs    = Numo::DFloat.ones(n_components, dim)  # diagonal variances
      end

      # X: Numo::DFloat[ T, D ]
      def fit(x, n_iter: 10)
        init_kmeans!(x)
        n_iter.times do
          log_resp = e_step(x)     # [T,K]
          resp = Numo::NMath.exp(log_resp)      # responsibilities γ_tk
          m_step(x, resp)
        end
      end

      # returns log p(x_t) as Numo::DFloat[T]
      def logpdf(x)
        t_len = x.shape[0]
        k = @n_components
        d = @dim
        x_e = x.reshape(t_len, 1, d)
        means_e = @means.reshape(1, k, d)
        vars_e = @covs.reshape(1, k, d)

        diff = x_e - means_e
        term1 = -0.5 * ((diff * diff) / vars_e).sum(2)
        comp_const = -0.5 * (::Math.log(2 * ::Math::PI) * d + Numo::NMath.log(@covs).sum(1))
        log_weights = Numo::NMath.log(@weights)

        log_comp = term1 + comp_const.reshape(1, k) + log_weights.reshape(1, k)
        m = log_comp.max(1)
        s = Numo::NMath.exp(log_comp - m.reshape(t_len, 1)).sum(1)
        m + Numo::NMath.log(s)
      end

      # Return a compact human-readable summary of the GMM (weights, means, covs)
      def summary
        parts = []
        parts << "GMM summary: comps=#{@n_components} dim=#{@dim}"
        @n_components.times do |k|
          w = @weights[k].is_a?(Numeric) ? @weights[k] : @weights[k].to_f
          mean = @means[k, true].to_a.map { |v| v.round(4) }
          cov = @covs[k, true].to_a.map { |v| v.round(4) }
          parts << " comp=#{k} weight=#{w.round(4)} mean=#{mean} cov=#{cov}"
        end
        parts.join("\n")
      end

      private

      def init_kmeans!(x, n_iter: 5)
        t_len = x.shape[0]
        idx = (0...t_len).to_a.sample(@n_components)
        @means = x[idx, true].dup

        n_iter.times do
          dists = pairwise_sq_dist(x, @means)        # [T,K]
          labels = dists.min_index(1)                # [T]
          @n_components.times do |k|
            mask = labels.eq(k)
            next if mask.where.empty?
            xk = x[mask, true]
            @means[k, true] = xk.mean(0)
          end
        end

        @covs = Numo::DFloat.ones(@n_components, @dim)
      end

      def e_step(x)
        t_len = x.shape[0]
        k = @n_components
        d = @dim

        x_e = x.reshape(t_len, 1, d)
        means_e = @means.reshape(1, k, d)
        vars_e = @covs.reshape(1, k, d)

        diff = x_e - means_e
        term1 = -0.5 * ((diff * diff) / vars_e).sum(2)
        comp_const = -0.5 * (::Math.log(2 * ::Math::PI) * d + Numo::NMath.log(@covs).sum(1))
        log_weights = Numo::NMath.log(@weights)

        log_resp = term1 + comp_const.reshape(1, k) + log_weights.reshape(1, k)

        # stable normalization per row
        max_row = log_resp.max(1)
        log_resp = log_resp - max_row.reshape(t_len, 1)
        log_norm = Numo::NMath.log( Numo::NMath.exp(log_resp).sum(1) )
        log_resp - log_norm.reshape(t_len, 1)
      end

      def m_step(x, resp)
        t_len = x.shape[0]
        nk = resp.sum(0) + 1e-8

        @weights = nk / t_len

        @n_components.times do |k|
          rk = resp[true, k].reshape(t_len, 1)
          num = (rk * x).sum(0)
          @means[k, true] = num / nk[k]
        end

        @n_components.times do |k|
          rk = resp[true, k].reshape(t_len, 1)
          diff = x - @means[k, true]
          num = (rk * diff * diff).sum(0)
          @covs[k, true] = num / nk[k] + 1e-6
        end
      end

      def log_gaussian(x, mean, var)
        diff = x - mean
        term1 = -0.5 * ((diff * diff) / var).sum
        term2 = -0.5 * (::Math.log(2 * ::Math::PI) * @dim +
                        Numo::NMath.log(var).sum)
        term1 + term2
      end

      def pairwise_sq_dist(x, centers)
        t_len = x.shape[0]
        k = centers.shape[0]
        x2 = (x * x).sum(1).reshape(t_len, 1)
        c2 = (centers * centers).sum(1).reshape(1, k)
        cross = x.dot(centers.transpose)
        out = x2 + c2 - 2 * cross
        out.clip(0, nil)
      end
    end
  end
end