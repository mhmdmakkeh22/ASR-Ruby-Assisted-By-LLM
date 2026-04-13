#!/usr/bin/env ruby
# bin/run_pipeline.rb
# End-to-end runner: lexicon -> LM -> cache features -> train -> (optional) word GMMs -> decode -> report

require 'fileutils'
require 'time'

PROJ = File.expand_path('..', __dir__)
MANIFEST = ENV['MANIFEST'] || File.join(PROJ, 'data', 'manifests', 'dev_manifest.tsv')
MODEL_OUT = ENV['MODEL_OUT'] || File.join(PROJ, 'exp', 'hmm_gmm_devclean', 'hmm_gmm_model.marshal')
WORD_GMM_OUT = ENV['WORD_GMM_OUT'] || File.join(PROJ, 'exp', 'word_gmms.marshal')

# training knobs (use ENV to override)
n_iter = ENV['N_ITER'] || '5'
em_iters = ENV['EM_ITERS_PER_STATE'] || '3'
components = ENV['N_COMPONENTS'] || '4'

log_dir = File.join(PROJ, 'exp')
FileUtils.mkdir_p(log_dir)
run_log = File.join(log_dir, "run_pipeline_#{Time.now.strftime('%Y%m%d_%H%M%S')}.log")

def sh(cmd)
  puts "> #{cmd}"
  start = Time.now
  ok = system(cmd)
  dur = Time.now - start
  puts "  -> exit=#{ok ? 0 : 1} time=#{dur.round(2)}s"
  return ok, dur
end

File.open(run_log, 'w') do |f|
  f.puts "Pipeline run: #{Time.now.iso8601}"
  f.puts "Manifest: #{MANIFEST}"
  f.puts "N_ITER=#{n_iter} EM_ITERS_PER_STATE=#{em_iters} N_COMPONENTS=#{components}"
end

# 1) Build lexicon
ok, t = sh("bundle exec ruby tools/build_lexicon.rb #{MANIFEST}")
abort("build_lexicon failed") unless ok
File.open(run_log, 'a') { |f| f.puts "build_lexicon: #{t.round(2)}s" }

# 2) Build unigram LM
ok, t = sh("bundle exec ruby tools/build_unigram_lm.rb #{MANIFEST}")
abort("build_unigram_lm failed") unless ok
File.open(run_log, 'a') { |f| f.puts "build_unigram_lm: #{t.round(2)}s" }

# 3) Cache features (fast if cache exists)
ok, t = sh("bundle exec ruby bin/cache_manifest_features.rb #{MANIFEST}")
abort("cache_manifest_features failed") unless ok
File.open(run_log, 'a') { |f| f.puts "cache_manifest_features: #{t.round(2)}s" }

# 4) Train HMM-GMM
train_cmd = "N_ITER=#{n_iter} EM_ITERS_PER_STATE=#{em_iters} bundle exec ruby bin/train_hmm_gmm.rb #{MANIFEST}"
ok, t = sh(train_cmd)
abort("train_hmm_gmm failed") unless ok
File.open(run_log, 'a') { |f| f.puts "train_hmm_gmm: #{t.round(2)}s" }

# 5) Optional: build per-word GMMs (can be slow). Skip if WORD_GMM_SKIP=1
unless ENV['WORD_GMM_SKIP'] == '1'
  # use smaller components and iterations by default for speed unless overridden
  wg_comp = ENV['WORD_GMM_COMPONENTS'] || components
  wg_iter = ENV['WORD_GMM_ITERS'] || '5'
  wg_min = ENV['WORD_GMM_MIN_FRAMES'] || '10'
  wg_max = ENV['WORD_GMM_MAX_UTTS'] || ''
  wg_args = [wg_comp, wg_iter, wg_min]
  wg_args << wg_max unless wg_max.to_s.strip.empty?
  ok, t = sh("bundle exec ruby tools/build_word_gmms.rb #{MANIFEST} #{WORD_GMM_OUT} #{wg_args.join(' ')}")
  if ok
    File.open(run_log, 'a') { |f| f.puts "build_word_gmms: #{t.round(2)}s" }
  else
    warn "build_word_gmms failed (continuing without per-word GMMs)"
    File.open(run_log, 'a') { |f| f.puts "build_word_gmms: FAILED" }
  end
end

# 6) Decode dev set with beam decoder and measure
beam_cmd = "bundle exec ruby bin/decoder_beam.rb #{MANIFEST} #{MANIFEST} 2 5 200 20 80 1.0 1.0 200 200"
ok, t = sh(beam_cmd)
File.open(run_log, 'a') { |f| f.puts "decoder_beam: #{ok ? 'OK' : 'FAIL'} time=#{t.round(2)}s" }

# 7) LL report
model_path = File.join(PROJ, 'exp', 'hmm_gmm_devclean', 'hmm_gmm_model.marshal')
ok, t = sh("bundle exec ruby bin/compute_ll_report.rb #{model_path} #{MANIFEST} #{MANIFEST} data/manifests/test_manifest.tsv")
File.open(run_log, 'a') { |f| f.puts "compute_ll_report: #{ok ? 'OK' : 'FAIL'} time=#{t.round(2)}s" }

puts "Pipeline finished. Run log: #{run_log}"

# exit with 0
exit 0
