#!/usr/bin/env ruby
# frozen_string_literal: true

require "optparse"
require "fileutils"

options = {
  sr: 16000,
  mono: false
}

OptionParser.new do |opts|
  opts.banner = "Convert LibriSpeech FLAC → WAV"

  opts.on("--in_root PATH", "Input LibriSpeech split root") do |v|
    options[:in_root] = v
  end

  opts.on("--out_root PATH", "Output WAV root") do |v|
    options[:out_root] = v
  end

  opts.on("--sr N", Integer, "Target sample rate (default 16000)") do |v|
    options[:sr] = v
  end

  opts.on("--mono", "Force mono") do
    options[:mono] = true
  end
end.parse!

unless options[:in_root] && options[:out_root]
  puts "Usage: ruby convert_librispeech_flac_to_wav.rb --in_root <path> --out_root <path> [--sr 16000] [--mono]"
  exit 1
end

in_root  = options[:in_root]
out_root = options[:out_root]
sr       = options[:sr]
mono     = options[:mono]

puts "Scanning FLAC files in: #{in_root}"

flac_files = Dir.glob(File.join(in_root, "**", "*.flac"))

if flac_files.empty?
  puts "No FLAC files found!"
  exit 1
end

puts "Found #{flac_files.size} files."

flac_files.each_with_index do |flac_path, idx|
  rel_path = flac_path.sub(/^#{Regexp.escape(in_root)}/, "")
  wav_path = File.join(out_root, rel_path).sub(/\.flac$/, ".wav")

  FileUtils.mkdir_p(File.dirname(wav_path))

  cmd = [
    "ffmpeg",
    "-y",
    "-i", "\"#{flac_path}\"",
    "-ar", sr.to_s,
    "-ac", mono ? "1" : "2",
    "-sample_fmt", "s16",
    "\"#{wav_path}\""
  ].join(" ")

  puts "[#{idx+1}/#{flac_files.size}] #{wav_path}"
  system(cmd)
end

puts "Conversion complete."