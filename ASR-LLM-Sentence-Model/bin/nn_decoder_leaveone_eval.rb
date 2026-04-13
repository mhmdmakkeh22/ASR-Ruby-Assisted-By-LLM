#!/usr/bin/env ruby
# bin/nn_decoder_leaveone_eval.rb
# k-NN sentence decoder leave-one-out evaluation on a manifest with transcripts

require 'numo/narray'
require 'json'

PROJ_ROOT = File.expand_path('..', __dir__)
CACHE_DIR = File.join(PROJ_ROOT, 'data', 'cache', 'features')
MANIFEST = ARGV[0] || File.join(PROJ_ROOT, 'data', 'manifests', 'dev_manifest.tsv')
K = (ARGV[1] || 1).to_i

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

rows = read_manifest(MANIFEST)
puts "Loading #{rows.length} entries from #{MANIFEST}"

ids = []
texts = []
vecs = []
rows.each do |r|
  id = r[:id]
  cache = File.join(CACHE_DIR, "#{id}.marshal")
  next unless File.exist?(cache)
  cached = Marshal.load(File.binread(cache))
  x = cached.is_a?(Hash) && cached.key?(:features) ? cached[:features] : cached
  arr = Numo::DFloat.cast(x)
  ids << id
  texts << r[:text]
  vecs << arr.mean(0)
end

puts "Built embeddings for #{ids.length} utterances"
train_mat = Numo::DFloat.cast(vecs.map { |v| v.to_a })

total_edits = 0
total_ref = 0
decoded = []

ids.each_with_index do |id, idx|
  q = train_mat[idx, true]
  sims = train_mat.dot(q.transpose).to_a
  # exclude self
  sims[idx] = -1e9
  # pick top K
  best = sims.each_with_index.sort_by { |s,i| -s }.first(K).map { |s,i| i }
  hyp = texts[best.first] || ""
  ref = texts[idx] || ""
  ref_toks = ref.split
  hyp_toks = hyp.split
  edits = levenshtein(ref_toks, hyp_toks)
  total_edits += edits
  total_ref += ref_toks.length
  decoded << { id: id, ref: ref, hyp: hyp, edits: edits }
  puts "#{idx+1}/#{ids.length} id=#{id} edits=#{edits}" if (idx % 500 == 0)
end

wer = total_ref > 0 ? (total_edits.to_f / total_ref) : nil
puts "\nLeave-one-out WER: #{wer.nil? ? 'N/A' : (wer * 100).round(2).to_s + '%'} (edits=#{total_edits} ref_words=#{total_ref})"
out = File.join(PROJ_ROOT, 'exp', 'nn_leaveone_results.json')
File.open(out, 'w') { |f| f.write(JSON.pretty_generate(decoded)) }
puts "Wrote results to #{out}"
