# bin/eval_hmm_gmm.rb
# Evaluate an existing HMM-GMM model on a manifest (uses cached features when available).

require 'fileutils'
require 'numo/narray'
require_relative '../lib/asr/math/hmm_gmm'

PROJ_ROOT = File.expand_path('..', __dir__)
CACHE_DIR = File.join(PROJ_ROOT, 'data', 'cache', 'features')
MODEL_PATH = ARGV[0] || File.join(PROJ_ROOT, 'exp', 'hmm_gmm_devclean', 'hmm_gmm_model.marshal')
MANIFEST   = ARGV[1] || File.join(PROJ_ROOT, 'data', 'manifests', 'test_manifest.tsv')

unless File.exist?(MODEL_PATH)
  abort "Model not found: #{MODEL_PATH}"
end

def read_manifest(path)
  rows = []
  File.foreach(path) do |line|
    line = line.strip
    next if line.empty?
    parts = line.split("\t")
    next if parts[0].downcase == 'id'
    id = parts[0]
    wav = parts.length >= 2 ? parts[1] : nil
    rows << { id: id, wav: wav }
  end
  rows
end

entries = read_manifest(MANIFEST)
puts "Manifest entries: #{entries.length}"

sequences = []
entries.each_with_index do |e, idx|
  id = e[:id]
  cache_path = File.join(CACHE_DIR, "#{id}.marshal")
  if File.exist?(cache_path)
    cached = Marshal.load(File.binread(cache_path))
    x = cached.is_a?(Hash) && cached.key?(:features) ? cached[:features] : cached
    sequences << x
  else
    warn "Missing cache for #{id}; skipping"
  end
end

if sequences.empty?
  abort "No sequences loaded from cache. Make sure features are computed and cached."
end

model = Marshal.load(File.binread(MODEL_PATH))
score = model.send(:score_sequences, sequences)
puts "Avg log-likelihood on #{sequences.length} sequences: #{score.round(4)}"

# Optional: print average frames per sequence
total_frames = sequences.inject(0) { |acc, x| acc + x.shape[0] }
puts "Total frames: #{total_frames}; avg frames/seq: #{(total_frames.to_f / sequences.length).round(2)}" 
