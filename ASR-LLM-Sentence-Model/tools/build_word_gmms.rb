#!/usr/bin/env ruby
# tools/build_word_gmms.rb
# Build per-word GMMs by assigning frames uniformly across words in each training utterance.

require 'fileutils'
require 'numo/narray'
require_relative '../lib/asr/math/gmm'

PROJ = File.expand_path('..', __dir__)
MANIFEST = ARGV[0] || File.join(PROJ, 'data', 'manifests', 'dev_manifest.tsv')
OUT_PATH = ARGV[1] || File.join(PROJ, 'exp', 'word_gmms.marshal')
N_COMPONENTS = (ARGV[2] || 4).to_i
N_ITER = (ARGV[3] || 5).to_i
MIN_FRAMES = (ARGV[4] || 10).to_i
MAX_UTTS = (ARGV[5] || nil)

puts "Reading manifest: #{MANIFEST}"

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

rows = read_manifest(MANIFEST)
cache_dir = File.join(PROJ, 'data', 'cache', 'features')

# optionally load forced-alignments if available
alignments_path = File.join(PROJ, 'exp', 'alignments.marshal')
alignments = File.exist?(alignments_path) ? Marshal.load(File.binread(alignments_path)) : {}

word_frames = Hash.new { |h,k| h[k] = [] }
count = 0
rows.each do |r|
  id = r[:id]
  break if MAX_UTTS && count >= MAX_UTTS.to_i
  cache = File.join(cache_dir, "#{id}.marshal")
  # prefer re-extracting features with VAD+CMVN for cleaner segmentation when wav available
  if File.exist?(cache)
    cached = Marshal.load(File.binread(cache))
    wav_path = cached.is_a?(Hash) ? cached[:wav] : nil
  else
    wav_path = nil
  end
  if wav_path && File.exist?(wav_path)
    # read raw and re-extract with VAD and CMVN
    require_relative '../lib/asr/audio/wav_reader'
    require_relative '../lib/asr/features/mfcc'
    audio = ASR::Audio::WavReader.read(wav_path)
    samples = audio[:samples]
    sr = audio[:sample_rate]
    mf = ASR::Features::MFCC.extract(samples, sr, { cmvn: :utterance, vad: { method: :energy, threshold_db: -40.0 } })
    x = mf[:data][:features]
    arr = Numo::DFloat.cast(x)
  elsif File.exist?(cache)
    cached = Marshal.load(File.binread(cache))
    x = cached.is_a?(Hash) && cached.key?(:features) ? cached[:features] : cached
    arr = Numo::DFloat.cast(x)
  else
    next
  end
  t = arr.shape[0]
  words = r[:text].split.map(&:downcase)
  next if words.empty?
  if alignments.key?(id)
    # use forced alignments when available
    segs = alignments[id]
    segs.each do |s|
      w = s[:word]
      start = s[:start]
      fin = s[:end]
      start = 0 if start < 0
      fin = t - 1 if fin >= t
      next if fin < start
      seg = arr[start..fin, true]
      seg.to_a.each { |row| word_frames[w] << row }
    end
  else
    # fallback: uniform segmentation across words
    per = (t.to_f / words.length)
    idx = 0
    words.each_with_index do |w, i|
      start = (i * per).floor
      fin = [t - 1, ((i + 1) * per).floor - 1].min
      fin = start if fin < start
      seg = arr[start..fin, true]
      seg.to_a.each { |row| word_frames[w] << row }
    end
  end
  count += 1
  puts "Processed #{count}/#{rows.length}" if (count % 200 == 0)
end

puts "Collected frames for #{word_frames.keys.length} words. Fitting GMMs..."
# If N_WORKERS set (>1), write per-word frame files and spawn worker processes (Windows-friendly)
n_workers = (ENV['N_WORKERS'] || '1').to_i
if n_workers > 1
  puts "Parallel mode: spawning #{n_workers} workers"
  frames_dir = File.join(PROJ, 'tmp', 'word_frames')
  FileUtils.rm_rf(frames_dir) if Dir.exist?(frames_dir)
  FileUtils.mkdir_p(frames_dir)
  # write per-word frame files
  word_frames.each do |w, frames|
    next if frames.length < MIN_FRAMES
    safe_name = w.gsub(/[^a-z0-9_-]/i, '_')
    File.open(File.join(frames_dir, "#{safe_name}.marshal"), 'wb') { |f| f.write(Marshal.dump({ word: w, frames: frames })) }
  end

  # spawn workers
  worker_outputs = []
  pids = []
  (0...n_workers).each do |wid|
    out = File.join(PROJ, 'exp', "word_gmms_worker_#{wid}.marshal")
    worker_outputs << out
    cmd = [Gem.ruby, File.join(PROJ, 'tools', 'build_word_gmms_worker.rb'), frames_dir, out, N_COMPONENTS.to_s, N_ITER.to_s, MIN_FRAMES.to_s, wid.to_s, n_workers.to_s]
    p = spawn(*cmd)
    pids << p
  end
  # wait for workers
  pids.each { |pid| Process.wait(pid) }

  # merge worker outputs
  merged = {}
  worker_outputs.each do |wp|
    next unless File.exist?(wp)
    begin
      h = Marshal.load(File.binread(wp))
      merged.merge!(h)
    rescue => _e
    end
  end
  FileUtils.mkdir_p(File.dirname(OUT_PATH))
  File.open(OUT_PATH, 'wb') { |f| f.write(Marshal.dump(merged)) }
  puts "Wrote #{merged.keys.length} GMMs to #{OUT_PATH} (merged from workers)"
else
  word_gmms = {}
  word_frames.each_with_index do |(w, frames), idx|
    begin
      if frames.length < MIN_FRAMES
        next
      end
      x = Numo::DFloat.cast(frames)
      g = ASR::Math::GMM.new(N_COMPONENTS, x.shape[1])
      g.fit(x, n_iter: N_ITER)
      word_gmms[w] = g
    rescue => ex
      warn "Failed to fit GMM for #{w}: #{ex.class}: #{ex.message}"
    end
    puts "Fitted #{idx+1}/#{word_frames.keys.length} words" if (idx % 200 == 0)
  end

  FileUtils.mkdir_p(File.dirname(OUT_PATH))
  File.open(OUT_PATH, 'wb') { |f| f.write(Marshal.dump(word_gmms)) }
  puts "Wrote #{word_gmms.keys.length} GMMs to #{OUT_PATH}"
end
