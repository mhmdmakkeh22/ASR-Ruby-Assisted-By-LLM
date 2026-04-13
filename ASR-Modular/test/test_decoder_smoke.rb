# frozen_string_literal: true

require "minitest/autorun"
require_relative "../lib/asr"

class TestDecoderSmoke < Minitest::Test
  def test_decoder_runs
    # Minimal smoke: build a tiny AM with a SIL phone only and decode silence features.
    dim = 13
    am = ASR::AM::HMMGMM.new(n_states: 5, n_components: 2, dim: dim)

    # Fake seed frames to init
    seed = Array.new(200) { Array.new(dim) { rand * 0.1 } }
    am.ensure_phone!("SIL", seed_data: seed)

    lex = ASR::Lexicon::LexiconStore.new({ "RED" => ["G_R"] })
    lm = ASR::LM.build_unigram(["RED"], add_one: 1.0)

    dec = ASR::Decoder::BeamSearch.new(am: am, lexicon: lex, lm: lm, beam_size: 20, max_words_per_sil: 5)

    feats = Array.new(30) { Array.new(dim) { rand * 0.1 } }
    out = dec.decode(feats)
    assert out.is_a?(Array)
  end
end
