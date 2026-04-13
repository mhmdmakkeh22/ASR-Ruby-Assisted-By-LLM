#!/usr/bin/env ruby
# tools/build_unigram_lm.rb
# Build a unigram LM (JSON of log-probs) from manifest transcripts

require 'json'

PROJ = File.expand_path('..', __dir__)
MANIFESTS = ARGV.empty? ? [File.join(PROJ, 'data', 'manifests', 'dev_manifest.tsv')] : ARGV
OUT = File.join(PROJ, 'data', 'unigram_lm.json')

counts = Hash.new(0)
total = 0
MANIFESTS.each do |m|
  next unless File.exist?(m)
  File.foreach(m) do |line|
    line = line.strip
    next if line.empty?
    parts = line.split("\t")
    next if parts[0].downcase == 'id'
    txt = parts[2] || ''
    words = txt.split.map(&:downcase)
    words.each { |w| counts[w] += 1; total += 1 }
  end
end

probs = {}
counts.each { |w,c| probs[w] = Math.log((c.to_f + 1.0) / (total + counts.length)) }

File.open(OUT, 'w') { |f| f.write(JSON.pretty_generate({ total_words: total, vocab_size: counts.length, log_probs: probs })) }
puts "Wrote unigram LM to #{OUT} (vocab=#{counts.length} total_words=#{total})"
