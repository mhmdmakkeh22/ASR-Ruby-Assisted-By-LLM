#!/usr/bin/env ruby
# bin/nn_sentence_decoder.rb
# Simple k-NN sentence-level decoder: index mean MFCC per training utterance
# and decode test utterances by nearest neighbor. Computes WER (Levenshtein).

require 'numo/narray'
require 'fileutils'
require 'json'

PROJ_ROOT = File.expand_path('..', __dir__)
CACHE_DIR = File.join(PROJ_ROOT, 'data', 'cache', 'features')
TRAIN_MAN = ARGV[0] || File.join(PROJ_ROOT, 'data', 'manifests', 'dev_manifest.tsv')
TEST_MAN  = ARGV[1] || File.join(PROJ_ROOT, 'data', 'manifests', 'test_manifest.tsv')
K = (ARGV[2] || 1).to_i

def read_manifest(path)
  rows = []
  File.foreach(path) do |line|
    line = line.strip
    next if line.empty?
    parts = line.split("\t")
    next if parts[0].downcase == 'id'
    id = parts[0]
    wav = parts.length >= 2 ? parts[1] : nil
    txt = parts.length >= 3 ? parts[2] : ""
    rows << { id: id, wav: wav, text: txt }
  end
  rows
end

def load_mean_embeddings(manifest)
  rows = read_manifest(manifest)
  ids = []
  texts = []
  vecs = []
  missing = 0
  rows.each do |r|
    id = r[:id]
    cache = File.join(CACHE_DIR, "#{id}.marshal")
    if File.exist?(cache)
      cached = Marshal.load(File.binread(cache))
      x = cached.is_a?(Hash) && cached.key?(:features) ? cached[:features] : cached
      arr = Numo::DFloat.cast(x)
      mean = arr.mean(0)
      vecs << mean
      ids << id
      texts << r[:text]
    else
      missing += 1
    end
  end
  if vecs.empty?
    return [[], [], [], missing]
  end
  # build matrix [N_train, D]
  mat = Numo::DFloat.cast(vecs.map { |v| v.to_a })
  [ids, texts, mat, missing]
end

def cosine_similarity_matrix(a, b)
  # a: [M,D], b: [N,D] -> [M,N]
  an = a / (a.abs.sum(1).reshape(a.shape[0],1) + 1e-12)
  bn = b / (b.abs.sum(1).reshape(b.shape[0],1) + 1e-12)
  an.dot(bn.transpose)
end

def levenshtein(a, b)
  # a,b arrays of tokens
  m = a.length
  n = b.length
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

puts "Building training index from #{TRAIN_MAN}"
train_ids, train_texts, train_mat, missing_train = load_mean_embeddings(TRAIN_MAN)
puts "  training rows: #{train_ids.length}; missing cached: #{missing_train}"

puts "Loading test set from #{TEST_MAN}"
test_rows = read_manifest(TEST_MAN)

total_edits = 0
total_ref_words = 0
correct_utterances = 0
decoded = []

test_rows.each_with_index do |r, idx|
  id = r[:id]
  cache = File.join(CACHE_DIR, "#{id}.marshal")
  unless File.exist?(cache)
    warn "skip #{id} missing cache"
    next
  end
  cached = Marshal.load(File.binread(cache))
  x = cached.is_a?(Hash) && cached.key?(:features) ? cached[:features] : cached
  arr = Numo::DFloat.cast(x)
  q = arr.mean(0)
  # compute cosine similarity to all training utterances
  sims = (train_mat.dot(q.transpose)).to_a
  # find top K indices
  idxs = sims.each_with_index.sort_by { |s,i| -s }.first(K).map { |s,i| i }
  hyp = train_texts[idxs.first] || ""
  ref = r[:text] || ""
  # compute WER
  ref_tokens = ref.split
  hyp_tokens = hyp.split
  edits = levenshtein(ref_tokens, hyp_tokens)
  total_edits += edits
  total_ref_words += ref_tokens.length
  correct_utterances += 1 if edits == 0
  decoded << { id: id, ref: ref, hyp: hyp, edits: edits }
  if (idx % 200 == 0)
    puts "decoded #{idx+1}/#{test_rows.length} id=#{id} edits=#{edits} ref_len=#{ref_tokens.length}"
  end
end

wer = total_ref_words > 0 ? (total_edits.to_f / total_ref_words) : nil
puts "\nDecoding finished. test utterances processed=#{correct_utterances}, total_ref_words=#{total_ref_words}, total_edits=#{total_edits}"
puts "Estimated WER (nn sentence-copy baseline): #{wer.nil? ? 'N/A' : (wer * 100).round(2).to_s + '%'}"

out_path = File.join(PROJ_ROOT, 'exp', 'nn_decoder_results.json')
File.open(out_path, 'w') { |f| f.write(JSON.pretty_generate(decoded)) }
puts "Wrote detailed results to #{out_path}"
