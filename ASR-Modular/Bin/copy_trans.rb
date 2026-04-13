#!/usr/bin/env ruby
# frozen_string_literal: true

require "fileutils"
require "optparse"

options = {}

OptionParser.new do |opts|
  opts.banner = "Copy .trans.txt files from LibriSpeech FLAC tree to WAV tree"

  opts.on("--in_root PATH", "Original FLAC root") { |v| options[:in_root] = v }
  opts.on("--out_root PATH", "Output WAV root")   { |v| options[:out_root] = v }
end.parse!

unless options[:in_root] && options[:out_root]
  puts "Usage: ruby copy_trans.rb --in_root <path> --out_root <path>"
  exit 1
end

# Normalize to forward slashes for Dir.glob compatibility on Windows
in_root  = options[:in_root].gsub("\\", "/")
out_root = options[:out_root].gsub("\\", "/")

trans_files = Dir.glob("#{in_root}/**/*.trans.txt")

if trans_files.empty?
  puts "No .trans.txt files found under #{in_root}"
  exit 1
end

puts "Found #{trans_files.size} .trans.txt file(s). Copying..."

trans_files.each do |src|
  rel = src.sub(/^#{Regexp.escape(in_root)}/, "")
  dst = out_root + rel

  FileUtils.mkdir_p(File.dirname(dst))
  FileUtils.cp(src, dst)
  puts "  #{dst}"
end

puts "Done."