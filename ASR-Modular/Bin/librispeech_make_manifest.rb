#!/usr/bin/env ruby
# frozen_string_literal: true

# Build a LibriSpeech manifest JSONL:
# Each line:
# {"id":"1272-128104-0000","audio_path":".../1272/128104/1272-128104-0000.wav","text":"...","words":["...","..."]}

require "json"
require "optparse"
require "fileutils"

def norm_path(p)
  # keep absolute + normalize slashes for cross-platform use
  File.expand_path(p.to_s).gsub("\\", "/")
end

def warn_puts(msg)
  $stderr.puts(msg)
end

opts = {
  librispeech_root: nil, # optional
  split: nil,
  audio_root: nil,
  audio_ext: "wav",
  out: nil,
  downcase: true
}

OptionParser.new do |o|
  o.banner = "Usage: ruby bin/librispeech_make_manifest.rb --split SPLIT --audio_root PATH --out PATH [--audio_ext wav] [--no_downcase]"

  o.on("--librispeech_root PATH", "Optional LibriSpeech root (not required if audio_root already points to LibriSpeech/<split>)") { |v| opts[:librispeech_root] = v }
  o.on("--split NAME", "Split folder name, e.g. train-clean-100, dev-clean, test-clean") { |v| opts[:split] = v }
  o.on("--audio_root PATH", "Path that contains LibriSpeech split folder (or directly the split folder). Examples: data/wav/dev-clean/LibriSpeech OR data/wav/dev-clean/LibriSpeech/dev-clean") { |v| opts[:audio_root] = v }
  o.on("--audio_ext EXT", "Audio extension (default: wav)") { |v| opts[:audio_ext] = v }
  o.on("--out PATH", "Output JSONL path") { |v| opts[:out] = v }
  o.on("--[no-]downcase", "Downcase transcripts (default: true)") { |v| opts[:downcase] = v }
end.parse!

if opts[:split].nil? || opts[:audio_root].nil? || opts[:out].nil?
  warn_puts("ERROR: need --split, --audio_root, --out")
  exit 1
end

audio_root = norm_path(opts[:audio_root])
split = opts[:split]
audio_ext = opts[:audio_ext].sub(/^\./, "")
out_path = norm_path(opts[:out])

# audio_root may already include /<split>. If not, append it.
split_root = if File.directory?(File.join(audio_root, split))
               norm_path(File.join(audio_root, split))
             else
               audio_root
             end

unless File.directory?(split_root)
  warn_puts("ERROR: split_root not found: #{split_root}")
  exit 1
end

trans_files = Dir.glob(File.join(split_root, "**", "*.trans.txt"))
if trans_files.empty?
  warn_puts("ERROR: no *.trans.txt found under: #{split_root}")
  exit 1
end

FileUtils.mkdir_p(File.dirname(out_path))

total = 0
missing_audio = 0

File.open(out_path, "w") do |out|
  trans_files.each do |tf|
    File.foreach(tf) do |line|
      line = line.strip
      next if line.empty?

      # format: <utt_id> <TRANSCRIPT...>
      parts = line.split(/\s+/, 2)
      next if parts.length < 2

      utt_id = parts[0]
      text = parts[1].strip
      text = text.downcase if opts[:downcase]

      # In LibriSpeech, the utt wav is usually alongside the .trans.txt in same folder:
      # .../<spk>/<chap>/<utt_id>.wav
      audio_path = norm_path(File.join(File.dirname(tf), "#{utt_id}.#{audio_ext}"))

      unless File.exist?(audio_path)
        missing_audio += 1
        warn_puts("WARN: missing audio for #{utt_id} expected #{audio_path}")
        next
      end

      words = text.gsub(/[^a-zA-Z'\s]/, " ").split(/\s+/).reject(&:empty?)
      rec = {
        id: utt_id,
        audio_path: audio_path,
        text: text,
        words: words
      }
      out.puts(JSON.generate(rec))
      total += 1
    end
  end
end

warn_puts("Done. Wrote #{total} utterances to #{out_path}")
warn_puts("Missing audio: #{missing_audio}") if missing_audio > 0