#!/usr/bin/env ruby
# tools/build_word_gmms_worker.rb
# Worker process: fits GMMs for a subset of per-word frame files.

require 'fileutils'
require 'numo/narray'

PROJ = File.expand_path('..', __dir__)
FRAMES_DIR = ARGV[0]
OUT_PATH = ARGV[1]
N_COMPONENTS = (ARGV[2] || 4).to_i
N_ITER = (ARGV[3] || 5).to_i
MIN_FRAMES = (ARGV[4] || 10).to_i
WORKER_ID = (ARGV[5] || 0).to_i
N_WORKERS = (ARGV[6] || 1).to_i

unless FRAMES_DIR && File.directory?(FRAMES_DIR)
  abort "frames_dir missing or invalid: #{FRAMES_DIR}"
end

require_relative '../lib/asr/math/gmm'

files = Dir.glob(File.join(FRAMES_DIR, '*.marshal')).sort
selected = files.each_with_index.select { |f, idx| (idx % N_WORKERS) == WORKER_ID }.map(&:first)

results = {}
selected.each_with_index do |f, idx|
  begin
    h = Marshal.load(File.binread(f))
    word = h.is_a?(Hash) && h.key?(:word) ? h[:word] : File.basename(f, '.marshal')
    frames = h.is_a?(Hash) && h.key?(:frames) ? h[:frames] : h
    next if frames.nil? || frames.length < MIN_FRAMES
    x = Numo::DFloat.cast(frames)
    g = ASR::Math::GMM.new(N_COMPONENTS, x.shape[1])
    g.fit(x, n_iter: N_ITER)
    results[word] = g
  rescue => ex
    warn "Worker #{WORKER_ID} failed on #{f}: #{ex.class}: #{ex.message}"
  end
  puts "Worker #{WORKER_ID}: processed #{idx+1}/#{selected.length}"
end

FileUtils.mkdir_p(File.dirname(OUT_PATH))
File.open(OUT_PATH, 'wb') { |fo| fo.write(Marshal.dump(results)) }
puts "Worker #{WORKER_ID} wrote #{results.keys.length} GMMs to #{OUT_PATH}"
