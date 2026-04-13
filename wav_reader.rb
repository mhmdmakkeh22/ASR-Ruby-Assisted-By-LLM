#PCM16 , little endian
#Goal: Load a .wav file (mono, 16-bit PCM) into normalized Float samples in [-1, 1]. For Phase 3 reliability, store your dataset already as 16 kHz mono PCM16 WAV.

# file: lib/asr/audio/wav_reader.rb
# Minimal WAV PCM16 LE reader (mono or stereo; converts to mono by averaging)
# Supports: RIFF/WAVE, fmt chunk, data chunk, PCM format (audio_format=1), 16-bit

module ASR
  module Audio
    class WavFormatError < StandardError; end

    class WavReader
      def self.read(path, target_sr: 16000, force_mono: true)
        File.open(path, "rb") do |f|
          riff = f.read(4)
          raise WavFormatError, "Not a RIFF file" unless riff == "RIFF"

          _riff_size = f.read(4) # unused
          wave = f.read(4)
          raise WavFormatError, "Not a WAVE file" unless wave == "WAVE"

          fmt = nil
          data = nil

          # Read chunks until we find "fmt " and "data"
          until f.eof?
            chunk_id = f.read(4)
            break if chunk_id.nil? || chunk_id.bytesize < 4

            chunk_size = f.read(4).unpack1("V")
            chunk_data_pos = f.pos

            case chunk_id
            when "fmt "
              raw = f.read(chunk_size)
              audio_format, num_channels, sample_rate, _byte_rate, _block_align, bits_per_sample =
                raw.unpack("v v V V v v")

              fmt = {
                audio_format: audio_format,
                num_channels: num_channels,
                sample_rate: sample_rate,
                bits_per_sample: bits_per_sample
              }
            when "data"
              data = f.read(chunk_size)
            else
              # skip unknown chunk
              f.seek(chunk_size, IO::SEEK_CUR)
            end

            # Chunks are word-aligned: if odd chunk_size, skip padding byte
            f.seek(chunk_data_pos + chunk_size + (chunk_size.odd? ? 1 : 0), IO::SEEK_SET)
          end

          raise WavFormatError, "Missing fmt chunk" if fmt.nil?
          raise WavFormatError, "Missing data chunk" if data.nil?
          raise WavFormatError, "Only PCM (format=1) supported" unless fmt[:audio_format] == 1
          raise WavFormatError, "Only 16-bit PCM supported" unless fmt[:bits_per_sample] == 16

          sr = fmt[:sample_rate]
          ch = fmt[:num_channels]

          # PCM16 little-endian signed
          ints = data.unpack("s<*") # array of signed 16-bit

          samples = if ch == 1
                      ints.map { |v| v / 32768.0 }
                    elsif ch == 2
                      # average L and R
                      mono = []
                      (0...(ints.length / 2)).each do |i|
                        l = ints[2*i] / 32768.0
                        rr = ints[2*i + 1] / 32768.0
                        mono << 0.5 * (l + rr)
                      end
                      mono
                    else
                      raise WavFormatError, "Unsupported channels: #{ch}"
                    end

          if sr != target_sr
            # For Phase 3 simplicity: prefer converting files to 16kHz offline.
            raise WavFormatError, "Sample rate #{sr} != target #{target_sr}. Convert WAV to #{target_sr}Hz first."
          end

          {
            data: { samples: samples, sr: sr },
            meta: { path: path, channels: ch, dtype: "pcm16", duration_s: samples.length.to_f / sr }
          }
        end
      end
    end
  end
end

