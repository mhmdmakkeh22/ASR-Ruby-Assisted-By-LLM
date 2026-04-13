#!/usr/bin/env ruby
# frozen_string_literal: true

require "json"
require "fileutils"

train_dir = "Data/wav/train-clean-100/LibriSpeech/train-clean-100"
output_file = "Data/manifests/train_clean_100.jsonl"

puts "Creating train-clean-100 manifest..."
puts "Source directory: #{train_dir}"
puts "Output file: #{output_file}"

# Find all speaker directories
speaker_dirs = Dir.glob(File.join(train_dir, "*")).select { |d| File.directory?(d) }
puts "Found #{speaker_dirs.length} speakers"

utterances = []

speaker_dirs.each_with_index do |speaker_dir, speaker_idx|
  speaker_id = File.basename(speaker_dir)
  puts "Processing speaker #{speaker_idx + 1}/#{speaker_dirs.length}: #{speaker_id}"
  
  # Find all chapter directories for this speaker
  chapter_dirs = Dir.glob(File.join(speaker_dir, "*")).select { |d| File.directory?(d) }
  
  chapter_dirs.each do |chapter_dir|
    chapter_id = File.basename(chapter_dir)
    
    # Find all wav files in this chapter
    wav_files = Dir.glob(File.join(chapter_dir, "*.wav"))
    
    # Try to find transcript file with speaker prefix
    speaker_id = File.basename(File.dirname(chapter_dir))  # e.g., "103"
    trans_file = File.join(chapter_dir, "#{speaker_id}-#{chapter_id}.trans.txt")
    
    if File.exist?(trans_file)
      # Read transcript file and find matching lines
      transcript_lines = File.readlines(trans_file, chomp: true)
      transcript_hash = {}
      
      transcript_lines.each do |line|
        if line.match(/^(\d+-\d+-\d+)\s+(.+)$/)
          utterance_id = $1
          transcript = $2.strip
          transcript_hash[utterance_id] = transcript
        end
      end
      
      wav_files.each do |wav_file|
        utterance_id = File.basename(wav_file, ".wav")
        
        if transcript_hash.key?(utterance_id)
          transcript = transcript_hash[utterance_id]
          
          # Create utterance entry
          utterance = {
            "id" => utterance_id,
            "audio_path" => File.absolute_path(wav_file),
            "text" => transcript,
            "words" => transcript.upcase.gsub(/[^A-Z0-9\s']/, " ").split
          }
          utterances << utterance
        else
          puts "  Warning: No transcript found for #{utterance_id}"
        end
      end
    else
      puts "  Warning: No transcript file #{trans_file}"
    end
  end
end

puts "Found #{utterances.length} utterances"

# Write manifest
puts "Writing manifest to #{output_file}"
File.open(output_file, "w") do |f|
  utterances.each { |utt| f.puts(JSON.generate(utt)) }
end

puts "Manifest created successfully!"
puts "Total utterances: #{utterances.length}"
puts "File size: #{File.size(output_file)} bytes"

# Show sample
puts "\nSample utterances:"
utterances.take(3).each_with_index do |utt, i|
  puts "  #{i+1}. #{utt['id']}: #{utt['text']}"
end
