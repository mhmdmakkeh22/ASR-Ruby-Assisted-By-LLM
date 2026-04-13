#!/usr/bin/env ruby
# bin/compute_ll_report.rb
# Compute per-sequence and per-frame average log-likelihoods for given manifests

require 'fileutils'
require 'numo/narray'
require_relative '../lib/asr/math/hmm_gmm'

PROJ_ROOT = File.expand_path('..', __dir__)
CACHE_DIR = File.join(PROJ_ROOT, 'data', 'cache', 'features')
MODEL_PATH = ARGV[0] || File.join(PROJ_ROOT, 'exp', 'hmm_gmm_devclean', 'hmm_gmm_model.marshal')
TRAIN_MAN = ARGV[1] || File.join(PROJ_ROOT, 'data', 'manifests', 'dev_manifest.tsv')
DEV_MAN   = ARGV[2] || File.join(PROJ_ROOT, 'data', 'manifests', 'dev_manifest.tsv')
TEST_MAN  = ARGV[3] || File.join(PROJ_ROOT, 'data', 'manifests', 'test_manifest.tsv')

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

def load_sequences_from_manifest(manifest)
  entries = read_manifest(manifest)
  seqs = []
  missing = 0
  entries.each do |e|
    id = e[:id]
    cache_path = File.join(CACHE_DIR, "#{id}.marshal")
    if File.exist?(cache_path)
      cached = Marshal.load(File.binread(cache_path))
      x = cached.is_a?(Hash) && cached.key?(:features) ? cached[:features] : cached
      seqs << Numo::DFloat.cast(x)
    else
      missing += 1
    end
  end
  [seqs, entries.length, missing]
end

unless File.exist?(MODEL_PATH)
  abort "Model not found: #{MODEL_PATH}"
end

puts "Loading model: #{MODEL_PATH}"
model = Marshal.load(File.binread(MODEL_PATH))

datasets = { "train" => TRAIN_MAN, "dev" => DEV_MAN, "test" => TEST_MAN }
report = {}

datasets.each do |name, man|
  puts "\nProcessing #{name} manifest: #{man}"
  seqs, total_entries, missing = load_sequences_from_manifest(man)
  puts "  entries listed: #{total_entries}; cached missing: #{missing}; loaded sequences: #{seqs.length}"
  if seqs.empty?
    puts "  no sequences loaded for #{name}; skipping"
    next
  end

  total_loglik = 0.0
  total_frames = 0
  seqs.each do |x|
    log_b = model.send(:emission_log_likelihoods, x)
    _la, log_l = model.send(:forward, log_b)
    total_loglik += log_l
    total_frames += x.shape[0]
  end

  avg_per_seq = total_loglik / seqs.length
  avg_per_frame = total_frames > 0 ? total_loglik / total_frames : 0.0
  report[name] = { entries: seqs.length, total_frames: total_frames, total_loglik: total_loglik, avg_per_seq: avg_per_seq, avg_per_frame: avg_per_frame }
  puts "  sequences: #{seqs.length}; total_frames: #{total_frames}"
  puts "  avg log-likelihood per sequence: #{avg_per_seq.round(4)}"
  puts "  avg log-likelihood per frame: #{avg_per_frame.round(6)}"
end

puts "\nSummary comparisons:"
datasets.keys.each do |k|
  r = report[k]
  if r
    puts "- #{k}: entries=#{r[:entries]} frames=#{r[:total_frames]} avg_seq=#{r[:avg_per_seq].round(4)} avg_frame=#{r[:avg_per_frame].round(6)}"
  else
    puts "- #{k}: no data"
  end
end
