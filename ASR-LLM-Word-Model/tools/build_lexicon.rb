#!/usr/bin/env ruby
# tools/build_lexicon.rb
# Build a simple lexicon (word -> word) from manifests' transcripts

PROJ = File.expand_path('..', __dir__)
MANIFESTS = ARGV.empty? ? [File.join(PROJ, 'data', 'manifests', 'dev_manifest.tsv')] : ARGV
OUT = File.join(PROJ, 'data', 'lexicon.txt')

words = {}
MANIFESTS.each do |m|
  next unless File.exist?(m)
  File.foreach(m) do |line|
    line = line.strip
    next if line.empty?
    parts = line.split("\t")
    next if parts[0].downcase == 'id'
    txt = parts[2] || ''
    txt.split.each { |w| words[w.downcase] = true }
  end
end

File.open(OUT, 'w') do |f|
  words.keys.sort.each do |w|
    f.puts "#{w} #{w}"
  end
end

puts "Wrote lexicon (#{words.length} words) to #{OUT}"
