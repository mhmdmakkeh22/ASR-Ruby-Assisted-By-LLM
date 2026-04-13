# frozen_string_literal: true

require "json"
require "fileutils"

module ASR
  module Utils
    EPS = 1e-12

    def self.ensure_dir(path)
      FileUtils.mkdir_p(path)
      path
    end

    def self.read_jsonl(path)
      rows = []
      File.foreach(path) do |line|
        line = line.strip
        next if line.empty?
        rows << JSON.parse(line)
      end
      rows
    end

    def self.write_jsonl(path, rows)
      ensure_dir(File.dirname(path))
      File.open(path, "w") do |f|
        rows.each { |r| f.puts(JSON.generate(r)) }
      end
      path
    end

    def self.json_load(path)
      JSON.parse(File.read(path))
    end

    def self.tokenize(text)
      # Uppercase; keep digits; turn everything else into spaces.
      text.to_s.upcase.gsub(/[^A-Z0-9\s']/i, " ").split.reject(&:empty?)
    end

    def self.log_sum_exp(a, b)
      # Stable log(exp(a)+exp(b)) for two args.
      return b if a == -Float::INFINITY
      return a if b == -Float::INFINITY
      m = (a > b) ? a : b
      m + Math.log(Math.exp(a - m) + Math.exp(b - m))
    end

    def self.logsumexp(arr)
      m = arr.max
      return -Float::INFINITY if m.nil? || m == -Float::INFINITY
      s = 0.0
      arr.each { |x| s += Math.exp(x - m) }
      m + Math.log(s)
    end

    def self.next_pow2(n)
      k = 1
      k <<= 1 while k < n
      k
    end

    def self.safe_log(x)
      Math.log([x, EPS].max)
    end

    # Write mono PCM16LE WAV
    def self.write_wav_pcm16le(path, samples, sr: 16_000)
      ensure_dir(File.dirname(path))
      pcm = samples.map do |v|
        v = [[v, -1.0].max, 1.0].min
        (v * 32767.0).round
      end.pack("s<*")

      fmt_chunk = [
        1,      # PCM
        1,      # channels
        sr,     # sample_rate
        sr * 2, # byte_rate
        2,      # block_align
        16      # bits_per_sample
      ].pack("v v V V v v")

      data_chunk = pcm
      riff_size = 4 + (8 + fmt_chunk.bytesize) + (8 + data_chunk.bytesize)

      File.open(path, "wb") do |f|
        f.write("RIFF")
        f.write([riff_size].pack("V"))
        f.write("WAVE")

        f.write("fmt ")
        f.write([fmt_chunk.bytesize].pack("V"))
        f.write(fmt_chunk)

        f.write("data")
        f.write([data_chunk.bytesize].pack("V"))
        f.write(data_chunk)
      end

      path
    end

    def self.marshal_dump(path, obj)
      ensure_dir(File.dirname(path))
      File.open(path, "wb") { |f| f.write(Marshal.dump(obj)) }
      path
    end

    def self.marshal_load(path)
      Marshal.load(File.binread(path))
    end

    def self.read_tsv(path)
      File.readlines(path, chomp: true).reject { |l| l.strip.empty? }.map { |l| l.split("\t") }
    end

    def self.write_lines(path, lines)
      ensure_dir(File.dirname(path))
      File.open(path, "w") { |f| lines.each { |l| f.puts(l) } }
      path
    end
  end
end
