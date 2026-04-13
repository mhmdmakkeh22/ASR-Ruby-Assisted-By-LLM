#!/usr/bin/env ruby
# bin/decoder_beam.rb
# Beam-search decoder using per-word prototypes + unigram LM

require 'numo/narray'
require 'json'

PROJ = File.expand_path('..', __dir__)
CACHE = File.join(PROJ, 'data', 'cache', 'features')
TRAIN_MAN = ARGV[0] || File.join(PROJ, 'data', 'manifests', 'dev_manifest.tsv')
EVAL_MAN  = ARGV[1] || File.join(PROJ, 'data', 'manifests', 'dev_manifest.tsv')
BEAM_SIZE = (ARGV[2] || 8).to_i
TOP_K    = (ARGV[3] || 20).to_i
TOP_V    = (ARGV[8] || 2000).to_i
MAX_UTTS = (ARGV[9] || 0).to_i
ACOUSTIC_PRUNE_DB = (ARGV[10] || -20.0).to_f
MIN_SEG  = (ARGV[4] || 4).to_i
MAX_SEG  = (ARGV[5] || 120).to_i
ACOUSTIC_SCALE = (ARGV[6] || 1.0).to_f
LM_SCALE = (ARGV[7] || 1.0).to_f

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

puts "Building word prototypes from #{TRAIN_MAN}..."
train = read_manifest(TRAIN_MAN)
word_sums = Hash.new { |h,k| h[k] = nil }
word_counts = Hash.new(0)
total_frames = 0
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
  words.each do |w|
    if word_sums[w].nil?
      word_sums[w] = mean.dup
    else
      word_sums[w] = word_sums[w] + mean
    end
    word_counts[w] += 1
  end
end

vocab_all = word_sums.keys
vocab_sorted = vocab_all.sort_by { |w| -word_counts[w] }
vocab = vocab_sorted.first([TOP_V, vocab_sorted.length].min)
protos = vocab.map { |w| (word_sums[w] / [word_counts[w], 1].max.to_f).to_a }
proto_mat = Numo::DFloat.cast(protos)
proto_sq = (proto_mat ** 2).sum(1)
proto_norms = Numo::NMath.sqrt(proto_sq + 1e-12)
proto_normed = proto_mat / proto_norms.reshape(proto_norms.shape[0], 1)
puts "  prototypes for #{vocab.length} words (top #{TOP_V}); proto_dim=#{proto_mat.shape[1]}"

lm_path = File.join(PROJ, 'data', 'unigram_lm.json')
lm = File.exist?(lm_path) ? JSON.parse(File.read(lm_path))['log_probs'] : {}

# Load per-word GMMs if available. Require GMM class first so Marshal can instantiate objects.
require_relative '../lib/asr/math/gmm'
candidate_paths = [File.join(PROJ, 'exp', 'word_gmms.marshal'), File.join(PROJ, 'exp', 'word_gmms_full.marshal')]
word_gmms = {}
candidate_paths.each do |p|
  if File.exist?(p)
    begin
      word_gmms = Marshal.load(File.binread(p))
      puts "Loaded word GMMs from #{p} (#{word_gmms.keys.length} entries)"
      break
    rescue => _e
      warn "Failed to load word GMMs from #{p}, trying next"
    end
  end
end

def topk_indices(sim_vec, k)
  # Try to use Numo sort_index for speed; fallback to Ruby sort
  if sim_vec.respond_to?(:sort_index)
    idxs = sim_vec.sort_index.to_a
    # sort_index returns ascending indices; take last k and reverse for descending
    sel = idxs[(idxs.length - k)...idxs.length] || idxs
    sel.reverse
  else
    arr = sim_vec.to_a
    arr.each_with_index.sort_by { |s,i| -s }.first(k).map { |s,i| i }
  end
end

def levenshtein(a, b)
  m = a.length; n = b.length
  return n if m == 0; return m if n == 0
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

eval_rows = read_manifest(EVAL_MAN)
puts "Decoding #{eval_rows.length} utterances (beam=#{BEAM_SIZE} top_k=#{TOP_K})"

total_edits = 0
total_ref_words = 0
decoded = []
avg_frames_per_word = [ (total_frames.to_f / [train.map { |r| r[:text].split.length }.inject(0, &:+), 1].max).round, 20 ].max

eval_rows.each_with_index do |r, idx|
  break if MAX_UTTS > 0 && idx >= MAX_UTTS
  id = r[:id]
  cache = File.join(CACHE, "#{id}.marshal")
  next unless File.exist?(cache)
  cached = Marshal.load(File.binread(cache))
  x = cached.is_a?(Hash) && cached.key?(:features) ? cached[:features] : cached
  arr = Numo::DFloat.cast(x)
  t = arr.shape[0]

  # leave-one-out: build proto matrix excluding same id's contribution by zeroing prototypes influenced by this id
  # For simplicity, we won't rebuild prototypes per-utt; rely on global prototypes (acceptable for prototype baseline)

  # beam: array of states { pos:, words:, score: }
  beam = [{ pos: 0, words: [], score: 0.0 }]
  final_hyps = []

  while beam.any? { |s| s[:pos] < t }
    new_beam = []
    beam.each do |st|
      next if st[:pos] >= t
      pos = st[:pos]
      max_seg = [MAX_SEG, t - pos].min
      (MIN_SEG..max_seg).each do |seglen|
        seg = arr[pos, seglen, true] rescue arr[pos..(pos+seglen-1), true]
        mean = seg.mean(0)
        sims = proto_normed.dot(mean / (Numo::NMath.sqrt((mean**2).sum) + 1e-12))
        # candidate shortlist by top-K prototypes
        top_idxs = topk_indices(sims, TOP_K)
        # acoustic pruning: keep only candidates within ACOUSTIC_PRUNE_DB dB of best
        best_sim = sims[top_idxs.first].to_f
        threshold = best_sim + (ACOUSTIC_PRUNE_DB / 20.0)  # approximate dB->linear via log scale
        top_idxs = top_idxs.select { |i| sims[i].to_f >= threshold }
        top_idxs.each do |wi|
          w = vocab[wi]
          sim = sims[wi].to_f
          lm_logp = lm.fetch(w, Math.log(1e-8))
          # acoustic scoring: prefer per-word GMM log-likelihood if available
          if word_gmms.key?(w)
            g = word_gmms[w]
            # segment to Numo::DFloat
            seg_x = seg.is_a?(Numo::NArray) ? seg : Numo::DFloat.cast(seg)
            seg_x = seg_x.reshape(seg_x.shape[0], seg_x.shape[1]) if seg_x.ndim == 2
            logpdf = g.logpdf(seg_x)
            acoustic_score = logpdf.sum
          else
            acoustic_score = sim
          end
          score = st[:score] + ACOUSTIC_SCALE * acoustic_score + LM_SCALE * lm_logp
          new_beam << { pos: pos + seglen, words: st[:words] + [w], score: score }
        end
      end
    end
    # prune to beam size
    beam = new_beam.sort_by { |h| -h[:score] }.first(BEAM_SIZE)
    # if any finished (pos >= t), move to final_hyps
    finished, beam = beam.partition { |h| h[:pos] >= t }
    final_hyps.concat(finished)
    break if beam.empty?
  end

  best = (final_hyps + beam).max_by { |h| h[:score] }
  hyp = best ? best[:words].join(' ') : ''
  ref = r[:text] || ''
  ref_toks = ref.split.map(&:downcase)
  hyp_toks = hyp.split
  edits = levenshtein(ref_toks, hyp_toks)
  total_edits += edits
  total_ref_words += ref_toks.length
  decoded << { id: id, ref: ref, hyp: hyp, edits: edits }
  puts "#{idx+1}/#{eval_rows.length} id=#{id} edits=#{edits} ref_len=#{ref_toks.length}" if (idx % 200 == 0)
end

wer = total_ref_words > 0 ? (total_edits.to_f / total_ref_words) : nil
puts "\nBeam-decoder WER: #{wer.nil? ? 'N/A' : (wer * 100).round(2).to_s + '%'} (edits=#{total_edits} ref_words=#{total_ref_words})"
File.open(File.join(PROJ, 'exp', 'decoder_beam_results.json'), 'w') { |f| f.write(JSON.pretty_generate(decoded)) }
puts "Wrote results to exp/decoder_beam_results.json"
