#!/usr/bin/env ruby
# tools/forced_align.rb
# Produce simple forced-alignments per utterance using trained HMM-GMM Viterbi paths.

require 'fileutils'
require 'numo/narray'

PROJ = File.expand_path('..', __dir__)
MANIFEST = ARGV[0] || File.join(PROJ, 'data', 'manifests', 'dev_manifest.tsv')
MODEL_PATH = ARGV[1] || File.join(PROJ, 'exp', 'hmm_gmm_devclean', 'hmm_gmm_model.marshal')
OUT_PATH = ARGV[2] || File.join(PROJ, 'exp', 'alignments.marshal')
CACHE_DIR = File.join(PROJ, 'data', 'cache', 'features')

def read_manifest(path)
  rows = []
  File.foreach(path) do |line|
    line = line.strip
    next if line.empty?
    parts = line.split("\t")
    next if parts[0].downcase == 'id'
    rows << { id: parts[0], wav: parts[1], text: (parts[2] || '') }
  end
  rows
end

puts "Loading HMM model: #{MODEL_PATH}"
model = nil
begin
  # ensure HMM/GMM classes are loaded so Marshal can deserialize
  require_relative '../lib/asr/math/hmm_gmm'
  require_relative '../lib/asr/math/gmm'
  model = Marshal.load(File.binread(MODEL_PATH))
rescue => ex
  abort "Failed to load model: #{ex.class}: #{ex.message}"
end

rows = read_manifest(MANIFEST)
alignments = {}
count = 0
rows.each do |r|
  id = r[:id]
  cache = File.join(CACHE_DIR, "#{id}.marshal")
  next unless File.exist?(cache)
  cached = Marshal.load(File.binread(cache))
  x = cached.is_a?(Hash) && cached.key?(:features) ? cached[:features] : cached
  arr = Numo::DFloat.cast(x)
  t = arr.shape[0]

  # viterbi returns state index per frame
  path = model.viterbi(arr)
  # compress runs of identical states into boundaries
  runs = []
  run_start = 0
  (1...path.length).each do |i|
    if path[i] != path[i-1]
      runs << { state: path[i-1], start: run_start, end: i - 1 }
      run_start = i
    end
  end
  runs << { state: path[-1], start: run_start, end: path.length - 1 }

  # distribute runs to words sequentially: create N word segments by grouping runs
  words = r[:text].split.map(&:downcase)
  if words.empty?
    count += 1
    next
  end
  nwords = words.length

  if runs.empty?
    # fallback to uniform segmentation when Viterbi produced no runs
    per = (t.to_f / nwords)
    word_segs = []
    words.each_with_index do |w, i|
      start = (i * per).floor
      fin = [t - 1, ((i + 1) * per).floor - 1].min
      fin = start if fin < start
      word_segs << { word: w, start: start, end: fin }
    end
  else
    # target number of runs per word (float)
    runs_per_word = runs.length.to_f / nwords
    word_segs = []
    ri = 0
    (0...nwords).each do |wi|
      # compute to index (inclusive) for this word
      to = [( ( (wi + 1) * runs_per_word ).round - 1 ), runs.length - 1].min
      to = ri if to < ri
      from = ri
      # guard indices
      from = 0 if from < 0
      # clamp from/to into valid range
      from = [from, runs.length - 1].min
      to = [to, runs.length - 1].min
      # ensure from <= to
      from = to if from > to
      seg_start = runs[from][:start]
      seg_end = runs[to][:end]
      word_segs << { word: words[wi], start: seg_start, end: seg_end }
      ri = to + 1
    end
    # if leftover runs, extend last segment
    if ri < runs.length && !word_segs.empty?
      word_segs[-1][:end] = runs[-1][:end]
    end
  end

  alignments[id] = word_segs
  count += 1
  puts "Aligned #{count}/#{rows.length} id=#{id} words=#{nwords} frames=#{t}" if (count % 200 == 0)
end

FileUtils.mkdir_p(File.dirname(OUT_PATH))
File.open(OUT_PATH, 'wb') { |f| f.write(Marshal.dump(alignments)) }
puts "Wrote alignments for #{alignments.keys.length} utts to #{OUT_PATH}"
