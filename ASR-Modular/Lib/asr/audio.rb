# frozen_string_literal: true

require "open3"

module ASR
  module Audio
    class AudioFormatError < StandardError; end

    # Returns: { samples: Array<Float>, sr: Integer }
    def self.read(path, target_sr: 16_000, force_mono: true, allow_ffmpeg_resample: true)
      ext = File.extname(path).downcase
      if ext == ".flac"
        return decode_via_ffmpeg(path, target_sr: target_sr, force_mono: force_mono)
      end

      # WAV (fast path)
      if ext == ".wav"
        begin
          wav = read_wav_pcm16le(path, target_sr: target_sr, force_mono: force_mono)
          return wav
        rescue AudioFormatError => e
          # If it's not RIFF/WAVE or SR mismatch and we allow ffmpeg, try ffmpeg.
          raise e unless allow_ffmpeg_resample
          return decode_via_ffmpeg(path, target_sr: target_sr, force_mono: force_mono)
        end
      end

      # Unknown extension: try sniffing header
      header = File.binread(path, 4)
      if header == "fLaC"
        decode_via_ffmpeg(path, target_sr: target_sr, force_mono: force_mono)
      elsif header == "RIFF"
        read_wav_pcm16le(path, target_sr: target_sr, force_mono: force_mono)
      else
        raise AudioFormatError, "Unsupported audio file: #{path} (need .wav PCM16LE or .flac)"
      end
    end

    def self.decode_via_ffmpeg(path, target_sr:, force_mono:)
      ac = force_mono ? "1" : "2"
      cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", path.to_s,
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ac", ac,
        "-ar", target_sr.to_s,
        "pipe:1"
      ]

      stdout, stderr, status = Open3.capture3(*cmd)
      unless status.success?
        raise AudioFormatError, "ffmpeg decode failed for #{path}: #{stderr.strip}"
      end

      stdout = stdout.dup
      stdout.force_encoding(Encoding::BINARY)

      ints = stdout.unpack("s<*")
      samples = ints.map { |v| v / 32768.0 }

      { samples: samples, sr: target_sr }
    end

    def self.read_wav_pcm16le(path, target_sr:, force_mono:)
      File.open(path, "rb") do |f|
        riff = f.read(4)
        raise AudioFormatError, "Not a RIFF file" unless riff == "RIFF"

        _riff_size = f.read(4)
        wave = f.read(4)
        raise AudioFormatError, "Not a WAVE file" unless wave == "WAVE"

        fmt = nil
        data = nil

        until f.eof?
          chunk_id = f.read(4)
          break if chunk_id.nil? || chunk_id.bytesize < 4
          chunk_size = f.read(4).unpack1("V")
          chunk_start = f.pos

          case chunk_id
          when "fmt "
            raw = f.read(chunk_size)
            # audio_format, channels, sample_rate, byte_rate, block_align, bits_per_sample
            audio_format, ch, sr, _br, _ba, bps = raw.unpack("v v V V v v")
            fmt = { audio_format: audio_format, ch: ch, sr: sr, bps: bps }
          when "data"
            data = f.read(chunk_size)
          else
            f.seek(chunk_size, IO::SEEK_CUR)
          end

          # word-align
          f.seek(chunk_start + chunk_size + (chunk_size.odd? ? 1 : 0), IO::SEEK_SET)
        end

        raise AudioFormatError, "Missing fmt chunk" if fmt.nil?
        raise AudioFormatError, "Missing data chunk" if data.nil?
        raise AudioFormatError, "Only PCM (format=1) supported" unless fmt[:audio_format] == 1
        raise AudioFormatError, "Only 16-bit PCM supported" unless fmt[:bps] == 16

        sr = fmt[:sr]
        raise AudioFormatError, "Sample rate #{sr} != target #{target_sr}" unless sr == target_sr

        ch = fmt[:ch]
        ints = data.unpack("s<*")

        samples =
          case ch
          when 1
            ints.map { |v| v / 32768.0 }
          when 2
            if force_mono
              mono = []
              (0...(ints.length / 2)).each do |i|
                l = ints[2 * i] / 32768.0
                r = ints[2 * i + 1] / 32768.0
                mono << 0.5 * (l + r)
              end
              mono
            else
              ints.map { |v| v / 32768.0 }
            end
          else
            raise AudioFormatError, "Unsupported channels: #{ch}"
          end

        { samples: samples, sr: sr }
      end
    end
  end
end
