#!/usr/bin/env ruby
# frozen_string_literal: true

require "optparse"
require_relative "../lib/asr"

opts = {
  manifest: nil,
  out: nil,
  config: "default",  # default, high_res, fast
  verbose: true
}

OptionParser.new do |o|
  o.banner = "Usage: ruby bin/extract_features_enhanced.rb --manifest MANIFEST --out OUTPUT_DIR [--config CONFIG]"
  o.on("--manifest PATH", "Manifest JSONL file") { |v| opts[:manifest] = v }
  o.on("--out DIR", "Output directory") { |v| opts[:out] = v }
  o.on("--config CONFIG", %w[default high_res fast], "Feature configuration") { |v| opts[:config] = v }
  o.on("--[no-]verbose", "Show progress") { |v| opts[:verbose] = v }
end.parse!

unless opts[:manifest] && opts[:out]
  puts "Error: --manifest and --out are required"
  exit 1
end

# Select configuration
case opts[:config]
when "high_res"
  config = ASR::Features::HIGH_RES_MFCC
  puts "Using HIGH-RES MFCC: 20 coefficients, 40 mel filters"
when "fast"
  config = ASR::Features::FAST_MFCC
  puts "Using FAST MFCC: 13 coefficients, 20 mel filters"
else
  config = ASR::Features::DEFAULT_MFCC
  puts "Using DEFAULT MFCC: 13 coefficients, 26 mel filters"
end

puts "Extracting features from #{opts[:manifest]} to #{opts[:out]}"
puts "Configuration: #{opts[:config]}"

# Load manifest
utterances = ASR::Utils.read_jsonl(opts[:manifest])
puts "Found #{utterances.length} utterances"

# Process each utterance
utterances.each_with_index do |utt, idx|
  if opts[:verbose] && (idx % 100 == 0 || idx == utterances.length - 1)
    progress = (idx + 1) * 100.0 / utterances.length
    puts "Progress: #{progress.round(1)}% (#{idx + 1}/#{utterances.length}) - #{utt['id']}"
  end

  # Read audio
  audio = ASR::Audio.read(utt["audio_path"])
  
  # Extract features with selected configuration
  feats = ASR::Features.mfcc(audio.samples, audio.sr, config)
  feats = ASR::Features.cmvn_utterance(feats)

  # Save features
  out_path = File.join(opts[:out], "#{utt['id']}.marshal")
  ASR::Utils.marshal_dump(out_path, {
    utt_id: utt["id"],
    text: utt["text"],
    words: utt["words"],
    feats: feats
  })
end

puts "Feature extraction completed!"
puts "Output directory: #{opts[:out]}"
puts "Configuration: #{opts[:config]}"
