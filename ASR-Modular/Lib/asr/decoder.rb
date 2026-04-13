# lib/asr/decoder.rb
# frozen_string_literal: true
require_relative "utils"
module ASR
  module Decoder
    History = Struct.new(:word, :prev, :len)
    Hyp = Struct.new(:score, :hist, :mode, :word, :phones, :phone_i, :state_i)

    def self.hist_push(hist, word)
      History.new(word, hist, (hist&.len || 0) + 1)
    end

    def self.hist_to_a(hist)
      arr = []
      h = hist
      while h
        arr << h.word
        h = h.prev
      end
      arr.reverse
    end

    class BeamSearch
      def initialize(am:, lexicon:, lm:, beam_size: 200, beam_delta: 20.0,
                     lm_weight: 0.8, word_penalty: -0.2, max_words_per_sil: 200)
        @am = am
        @lexicon = lexicon
        @lm = lm
        @beam_size = beam_size
        @beam_delta = beam_delta
        @lm_weight = lm_weight
        @word_penalty = word_penalty
        @max_words_per_sil = max_words_per_sil

        # CHANGE: SIL missing is a fatal config error for this decoder.
        raise ArgumentError, "AM missing phone SIL" unless @am.has_phone?("SIL")

        # CHANGE: filter vocab to words whose phones exist in AM (prevents KeyError).
        raw = @lexicon.vocab.reject { |w| w.to_s.upcase == "<SIL>" || w.to_s.upcase == "SIL" }
        kept = []
        dropped = []
        raw.each do |w|
          phs = @lexicon.phones_for(w)
          (phs && phs.all? { |ph| @am.has_phone?(ph) }) ? kept << w : dropped << w
        end
        @vocab = kept.sort
        @vocab_sorted = @vocab.sort_by { |w| -@lm.log_prob(w) }
        if dropped.any?
          ex = dropped.first(5).join(", ")
          warn "Decoder vocab filtered: kept=#{@vocab.length}/#{raw.length}, dropped=#{dropped.length} (e.g., #{ex})"
        end
      end

      def decode(feats)
        t_total = feats.length
        return [] if t_total == 0
        h0 = Hyp.new(
          0.0 + @am.log_emit_safe("SIL", 0, feats[0]),
          nil,
          :sil,
          nil,
          ["SIL"],
          0,
          0
        )
        hyps = [h0]
        (0...(t_total - 1)).each do |t|
          x_next = feats[t + 1]
          next_map = {}
          hyps.each do |h|
            ph = h.phones[h.phone_i]
            st = h.state_i

            # self transition (safe; if invalid => -inf and pruned)
            ls = @am.log_self_safe(ph, st)
            le = @am.log_emit_safe(ph, st, x_next)
            if ls.finite? && le.finite?
              add_hyp(next_map, h, h.score + ls + le, h.mode, h.word, h.phones, h.phone_i, h.state_i)
            end

            if st < (@am.n_states - 1)
              ln = @am.log_next_safe(ph, st)
              le2 = @am.log_emit_safe(ph, st + 1, x_next)
              if ln.finite? && le2.finite?
                add_hyp(next_map, h, h.score + ln + le2, h.mode, h.word, h.phones, h.phone_i, st + 1)
              end
            else
              if h.phone_i < (h.phones.length - 1)
                ph2 = h.phones[h.phone_i + 1]
                ln = @am.log_next_safe(ph, st)
                le2 = @am.log_emit_safe(ph2, 0, x_next)
                if ln.finite? && le2.finite?
                  add_hyp(next_map, h, h.score + ln + le2, h.mode, h.word, h.phones, h.phone_i + 1, 0)
                end
              else
                if h.mode == :word
                  new_hist = ASR::Decoder.hist_push(h.hist, h.word)
                  ln = @am.log_next_safe(ph, st)
                  le2 = @am.log_emit_safe("SIL", 0, x_next)
                  if ln.finite? && le2.finite?
                    add_hyp(next_map, h, h.score + ln + le2, :sil, nil, ["SIL"], 0, 0, hist_override: new_hist)
                  end
                else
                  # branch from SIL into a word
                  @vocab_sorted.first(@max_words_per_sil).each do |w|
                    w_phones = @lexicon.phones_for(w)
                    next if w_phones.nil? || w_phones.empty?
                    ph2 = w_phones[0]
                    ln = @am.log_next_safe(ph, st)
                    le2 = @am.log_emit_safe(ph2, 0, x_next)
                    next unless ln.finite? && le2.finite?
                    lm_bonus = @lm_weight * @lm.log_prob(w) + @word_penalty
                    add_hyp(next_map, h, h.score + ln + lm_bonus + le2, :word, w, w_phones, 0, 0)
                  end
                end
              end
            end
          end
          hyps = next_map.values
          best = hyps.map(&:score).max || -Float::INFINITY
          cutoff = best - @beam_delta
          hyps.select! { |hh| hh.score >= cutoff }
          hyps.sort_by! { |hh| -hh.score }
          hyps = hyps.first(@beam_size)
        end
        best_sil = hyps.select { |h| h.mode == :sil }.max_by(&:score)
        best_any = hyps.max_by(&:score)
        best = best_sil || best_any
        out = ASR::Decoder.hist_to_a(best.hist)
        out << best.word if best.mode == :word && best.word
        out
      end

      private

      def add_hyp(map, prev_h, score, mode, word, phones, phone_i, state_i, hist_override: nil)
        key = [mode, word, phone_i, state_i].join("|")
        hist = hist_override || prev_h.hist
        cand = Hyp.new(score, hist, mode, word, phones, phone_i, state_i)
        cur = map[key]
        map[key] = cand if cur.nil? || cand.score > cur.score
      end
    end
  end
end