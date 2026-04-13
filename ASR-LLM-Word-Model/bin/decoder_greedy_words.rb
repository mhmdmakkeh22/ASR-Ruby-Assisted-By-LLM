#!/usr/bin/env ruby
# bin/decoder_greedy_words.rb
# Greedy word-level decoder using fixed-length segmentation and word prototypes

require 'numo/narray'
require 'json'

PROJ = File.expand_path('..', __dir__)
CACHE = File.join(PROJ, 'data', 'cache', 'features')
LEXICON = File.join(PROJ, 'data', 'lexicon.txt')
LM = File.join(PROJ, 'data', 'unigram_lm.json')
TRAIN_MAN = ARGV[0] || File.join(PROJ, 'data', 'manifests', 'dev_manifest.tsv')
TEST_MAN  = ARGV[1] || File.join(PROJ, 'data', 'manifests', 'test_manifest.tsv')

def read_manifest(path)
  rows = []
  File.foreach(path) do |line|
    line = line.strip
    next if line.empty?
    parts = line.split("\t")
    next if parts[0].downcase == 'id'
    id = parts[0]
    wav = parts[1]
    txt = parts[2] || ''
    rows << { id: id, wav: wav, text: txt }
  end
  rows
end

puts "Building lexicon and LM if missing..."
unless File.exist?(LEXICON)
  system("ruby", File.join(PROJ, 'tools', 'build_lexicon.rb'), TRAIN_MAN)
end
unless File.exist?(LM)
  system("ruby", File.join(PROJ, 'tools', 'build_unigram_lm.rb'), TRAIN_MAN)
end
lm = JSON.parse(File.read(LM)) rescue {}
log_probs = lm['log_probs'] || {}

train = read_manifest(TRAIN_MAN)
puts "Loading training prototypes from cache (crude per-word prototypes)..."
word_sums = Hash.new { |h,k| h[k] = Numo::DFloat.zeros(0) }
word_counts = Hash.new(0)
total_frames = 0
total_words = 0
train.each do |r|
  id = r[:id]
  cache = File.join(CACHE, "#{id}.marshal")
  next unless File.exist?(cache)
  cached = Marshal.load(File.binread(cache))
  x = cached.is_a?(Hash) && cached.key?(:features) ? cached[:features] : cached
  arr = Numo::DFloat.cast(x)
  mean = arr.mean(0)
  words = r[:text].split.map(&:downcase)
  total_frames += arr.shape[0]
  total_words += words.length
  words.each do |w|
    if word_sums[w].size == 0
      word_sums[w] = mean.dup
    else
      word_sums[w] = word_sums[w] + mean
    end
    word_counts[w] += 1
  end
end

vocab = word_sums.keys.sort
prototypes = {}
vocab.each do |w|
  prototypes[w] = (word_sums[w] / word_counts[w].to_f)
end
avg_frames_per_word = (total_words > 0) ? (total_frames.to_f / total_words).round : 20
puts "Built prototypes for #{vocab.length} words; avg_frames_per_word=#{avg_frames_per_word}"

test = read_manifest(TEST_MAN)
decoded = []
total_edits = 0
total_ref_words = 0
require_relative '../bin/nn_sentence_decoder' rescue nil

def levenshtein(a, b)
  m = a.length; n = b.length
  return n if m == 0
  return m if n == 0
  dp = Array.new(m+1) { Array.new(n+1, 0) }
  (0..m).each { |i| dp[i][0] = i }
  (0..n).each { |j| dp[0][j] = j }
  (1..m).each do |i|
    (1..n).each do |j|
      cost = (a[i-1] == b[j-1]) ? 0 : 1
      dp[i][j] = [dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost].min
    end
  end
  dp[m][n]
end

test.each_with_index do |r, idx|
  id = r[:id]
  cache = File.join(CACHE, "#{id}.marshal")
  unless File.exist?(cache)
    warn "skip #{id} missing cache"
    next
  end
  cached = Marshal.load(File.binread(cache))
  x = cached.is_a?(Hash) && cached.key?(:features) ? cached[:features] : cached
  arr = Numo::DFloat.cast(x)
  t = arr.shape[0]
  seg_len = [1, avg_frames_per_word].max
  nseg = (t.to_f / seg_len).ceil
  words = []
  (0...nseg).each do |s|
    start = s * seg_len
    fin = [t - 1, (start + seg_len - 1)].min
    seg = arr[start..fin, true]
    mean = seg.mean(0)
    # find nearest prototype by cosine
    best_w = nil
    best_sim = -1e9
    prototypes.each do |w, proto|
      sim = (mean.dot(proto.transpose)).to_f
      if sim > best_sim
        best_sim = sim
        best_w = w
      end
    end
    words << (best_w || '<unk>')
  end
  hyp = words.join(' ')
  ref = r[:text] || ''
  ref_toks = ref.split.map(&:downcase)
  hyp_toks = hyp.split
  edits = levenshtein(ref_toks, hyp_toks)
  total_edits += edits
  total_ref_words += ref_toks.length
  decoded << { id: id, ref: ref, hyp: hyp, edits: edits }
  puts "decoded #{idx+1}/#{test.length} id=#{id} edits=#{edits}" if (idx % 200 == 0)
end

wer = total_ref_words > 0 ? (total_edits.to_f / total_ref_words) : nil
puts "\nGreedy word-decoder WER: #{wer.nil? ? 'N/A' : (wer * 100).round(2).to_s + '%'} (edits=#{total_edits} ref_words=#{total_ref_words})"
File.open(File.join(PROJ, 'exp', 'decoder_greedy_results.json'), 'w') { |f| f.write(JSON.pretty_generate(decoded)) }
puts "Wrote results to exp/decoder_greedy_results.json"
