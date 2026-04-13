#Goal: Convert 1D samples into overlapping frames for short-time analysis. For sr=16000: frame_len=400 (25ms), hop_len=160 (10ms).

# file: lib/asr/preprocess/framing.rb
module ASR
  module Preprocess
    class Framing
      def self.frame(samples, sr:, frame_ms: 25.0, hop_ms: 10.0)
        frame_len = (sr * frame_ms / 1000.0).round
        hop_len   = (sr * hop_ms / 1000.0).round

        raise ArgumentError, "frame_len must be > 0" if frame_len <= 0
        raise ArgumentError, "hop_len must be > 0" if hop_len <= 0
        raise ArgumentError, "samples empty" if samples.nil? || samples.empty?

        frames = []
        i = 0
        while i + frame_len <= samples.length
          frames << samples[i, frame_len]
          i += hop_len
        end

        {
          data: { frames: frames, sr: sr, frame_len: frame_len, hop_len: hop_len },
          meta: { frame_ms: frame_ms, hop_ms: hop_ms, num_frames: frames.length }
        }
      end
    end
  end
end

