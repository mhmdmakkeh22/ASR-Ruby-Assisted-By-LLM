# frozen_string_literal: true

module ASR
  module MathUtil
    # Stable log-sum-exp for an array of log-values.
    # logsumexp([a,b,c]) = m + log(exp(a-m)+exp(b-m)+exp(c-m))
    def self.logsumexp(log_vals)
      raise ArgumentError, "empty log_vals" if log_vals.nil? || log_vals.empty?
      m = log_vals.max
      s = 0.0
      log_vals.each { |v| s += ::Math.exp(v - m) }
      m + ::Math.log(s)
    end
  end
end

