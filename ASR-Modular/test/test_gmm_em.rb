# frozen_string_literal: true

require "minitest/autorun"
require_relative "../lib/asr"

class TestGMM < Minitest::Test
  def test_gmm_logpdf_finite
    data = []
    200.times { data << [rand * 0.2 + 1.0, rand * 0.2 + 2.0] }
    g = ASR::AM::DiagGMM.from_data(data, k: 2, var_floor: 1e-3)
    ll = data.map { |x| g.log_pdf(x) }.sum / data.length.to_f
    assert ll.finite?
  end
end
