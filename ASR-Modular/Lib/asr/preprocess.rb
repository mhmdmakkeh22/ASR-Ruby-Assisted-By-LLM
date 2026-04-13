# frozen_string_literal: true

module ASR
  module Preprocess
    def self.normalize_peak(x)
      peak = x.map(&:abs).max.to_f
      return x.dup if peak <= 0.0
      x.map { |v| v / peak }
    end

    def self.pre_emphasis(x, coef: 0.97)
      return [] if x.empty?
      y = Array.new(x.length, 0.0)
      y[0] = x[0].to_f
      (1...x.length).each do |i|
        y[i] = x[i].to_f - coef * x[i - 1].to_f
      end
      y
    end

    def self.frame_signal(x, sr:, win_len_ms: 25.0, hop_ms: 10.0, pad: true)
      win_len = (sr * win_len_ms / 1000.0).round
      hop = (sr * hop_ms / 1000.0).round
      raise ArgumentError, "win_len must be > 0" if win_len <= 0
      raise ArgumentError, "hop must be > 0" if hop <= 0

      frames = []
      i = 0
      while i < x.length
        frame = x[i, win_len] || []
        if frame.length < win_len
          break unless pad
          frame = frame + Array.new(win_len - frame.length, 0.0)
        end
        frames << frame
        i += hop
        break if !pad && (i + win_len > x.length)
      end
      frames
    end

    def self.hamming(n)
      return [] if n <= 0
      return [1.0] if n == 1
      (0...n).map { |i| 0.54 - 0.46 * Math.cos(2.0 * Math::PI * i / (n - 1)) }
    end

    def self.apply_window(frames, window)
      frames.map do |fr|
        fr.zip(window).map { |a, w| a.to_f * w.to_f }
      end
    end
  end
end
