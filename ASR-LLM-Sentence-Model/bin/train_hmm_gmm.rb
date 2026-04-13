# bin/train_hmm_gmm.rb
# Train a simple HMM-GMM acoustic model using MFCC features.

require "csv"
require "fileutils"
require "numo/narray"

# ---- project libs ----
require_relative "../lib/asr/audio/wav_reader"
require_relative "../lib/asr/preprocess/normalize"
require_relative "../lib/asr/features/mfcc"
require_relative "../lib/asr/math/hmm_gmm"

# -----------------------------
# CONFIG (EDIT THESE PATHS)
# -----------------------------
PROJ_ROOT = File.expand_path("..", __dir__)

# Your WAV root (contains subfolders like 2277/149897/xxx.wav)
WAV_ROOT  = File.join(PROJ_ROOT, "data", "wav", "dev-clean")
CACHE_DIR = File.join(PROJ_ROOT, "data", "cache", "features")
FileUtils.mkdir_p(CACHE_DIR)

# Manifest should be TSV with either:
#   id<TAB>wav_path<TAB>text
# or
#   id<TAB>text
MANIFEST  = File.join(PROJ_ROOT, "data", "manifests", "dev_manifest.tsv")

OUT_DIR   = File.join(PROJ_ROOT, "exp", "hmm_gmm_devclean")
FileUtils.mkdir_p(OUT_DIR)

# MFCC params (keep consistent across train/test)
SAMPLE_RATE = 16_000

# HMM-GMM params
N_STATES     = 5
N_COMPONENTS = 4
N_ITER       = (ENV['N_ITER'] || '5').to_i
EM_ITERS_PER_STATE = (ENV['EM_ITERS_PER_STATE'] || '3').to_i

# -----------------------------
# Helpers
# -----------------------------
def read_manifest(path)
  rows = []
  File.foreach(path) do |line|
    line = line.strip
    next if line.empty?
    parts = line.split("\t")
    # Skip header rows like: id\twav_path\ttext
    if parts[0].downcase == 'id' || parts[0].downcase == 'index'
      next
    end

    if parts.length >= 3
      id, wav_path, text = parts[0], parts[1], parts[2..].join("\t")
      rows << { id: id, wav: wav_path, text: text }
    elsif parts.length == 2
      id, text = parts
      rows << { id: id, wav: nil, text: text }
    else
      warn "SKIP manifest line (bad format): #{line}"
    end
  end
  rows
end

def wav_path_from_id(wav_root, id)
  # If you saved wavs inside nested folders, you MUST have wav path in manifest.
  # This fallback assumes a flat layout: WAV_ROOT/<id>.wav
  File.join(wav_root, "#{id}.wav")
end

def extract_mfcc_from_wav(wav_path)
  audio = ASR::Audio::WavReader.read(wav_path)
  samples = audio[:samples]
  sr      = audio[:sample_rate]

  # normalize to [-1,1] peak
  samples = ASR::Preprocess::Normalize.peak(samples)

  mfcc_out = ASR::Features::MFCC.extract(samples, sr)
  # MFCC.extract returns a hash with :data => { features: [...] }
  feats = mfcc_out[:data][:features]

  # feats is Array<Array<Float>> (T x D)
  x = Numo::DFloat.cast(feats)
  x = x.reshape(feats.length, feats.first.length)
  x
end

# -----------------------------
# MAIN
# -----------------------------
puts "Reading manifest: #{MANIFEST}"
entries = read_manifest(MANIFEST)
puts "Manifest entries: #{entries.length}"

sequences = []

rebuild_cache = ENV['REBUILD_CACHE'] == '1'
print_every = [ (entries.length / 40), 1 ].max

entries.each_with_index do |e, idx|
  id = e[:id]
  wav = e[:wav] || wav_path_from_id(WAV_ROOT, id)

  begin
    # If manifest points to a non-existent file (often .flac), try fallback to converted WAVs
    unless File.exist?(wav)
      basename = File.basename(wav).sub(/\.flac$/i, '.wav')
      matches = Dir.glob(File.join(WAV_ROOT, "**", basename))
      if matches && !matches.empty?
        wav = matches.first
      else
        warn "SKIP id=#{id} (wav not found): #{wav}"
        next
      end
    end

    cache_path = File.join(CACHE_DIR, "#{id}.marshal")
    if !rebuild_cache && File.exist?(cache_path)
      # load cached features (support new { features:, t:, d: } format and older raw features)
      cached = Marshal.load(File.binread(cache_path))
      if cached.is_a?(Hash) && cached.key?(:features)
        x = cached[:features]
        if cached.key?(:t) && cached.key?(:d)
          t = cached[:t]
          d = cached[:d]
        else
          t = x.shape[0]
          d = x.shape[1]
        end
      else
        # legacy: cached directly contains Numo array
        x = cached
        t = x.shape[0]
        d = x.shape[1]
      end
      cached_from = :cache
    else
      x = extract_mfcc_from_wav(wav)
      if x.shape[0] < 2
        warn "SKIP id=#{id} (too short after MFCC): T=#{x.shape[0]}"
        next
      end
      # save cache
      begin
        File.open(cache_path, "wb") { |f| f.write(Marshal.dump({ features: x, t: x.shape[0], d: x.shape[1], wav: wav })) }
      rescue => _e
        # non-fatal if cache save fails
      end
      t = x.shape[0]
      d = x.shape[1]
      cached_from = :computed
    end

    sequences << x

    # concise progress: print every few files (approx 40 updates) with percent
    if (idx % print_every == 0) || idx == entries.length - 1
      pct = ((idx + 1).to_f / entries.length * 100).round(1)
      puts "Progress: #{pct}% (#{idx+1}/#{entries.length}) — id=#{id} T=#{t} D=#{d} [#{cached_from}]"
    end
  rescue => ex
    warn "SKIP id=#{id} wav=#{wav}"
    warn "  error: #{ex.class}: #{ex.message}"
  end
end

if sequences.empty?
  abort "No valid sequences found. Check manifest wav paths + wav conversion."
end

feature_dim = sequences.first.shape[1]
puts "Training HMM-GMM: states=#{N_STATES}, comps=#{N_COMPONENTS}, dim=#{feature_dim}"
model = ASR::Math::HMMGMM.new(N_STATES, feature_dim, n_components: N_COMPONENTS)

puts "Training config: N_ITER=#{N_ITER}, EM_ITERS_PER_STATE=#{EM_ITERS_PER_STATE}, N_STATES=#{N_STATES}, N_COMPONENTS=#{N_COMPONENTS}"
STDOUT.flush

# Optional dev split (fraction between 0 and 1). Set via ENV DEV_SPLIT (e.g. 0.1)
dev_split = (ENV['DEV_SPLIT'] || 0.0).to_f
dev_sequences = nil
if dev_split > 0 && dev_split < 1
  # random sample dev sequences (keep reproducible)
  rng = Random.new(1234)
  idxs = (0...sequences.length).to_a.shuffle(random: rng)
  k = (sequences.length * dev_split).round
  dev_idxs = idxs[0...k]
  dev_sequences = dev_idxs.map { |i| sequences[i] }
  # train_sequences = sequences - dev
  train_sequences = idxs[k...sequences.length].map { |i| sequences[i] }
else
  train_sequences = sequences
end

overall_start = Time.now
puts "Training started at #{overall_start.strftime('%Y-%m-%d %H:%M:%S')}"
STDOUT.flush

# call train with logger and checkpointing to OUT_DIR
log_path = File.join(OUT_DIR, "train.log")
model.train(train_sequences, n_iter: N_ITER, em_iters_per_state: EM_ITERS_PER_STATE, logger: log_path, checkpoint_dir: OUT_DIR, dev_sequences: dev_sequences)

overall_end = Time.now
overall_dur = overall_end - overall_start
puts "Training finished at #{overall_end.strftime('%Y-%m-%d %H:%M:%S')} (total duration: #{overall_dur.round(2)}s)"
STDOUT.flush

# Save model (simple Ruby Marshal)
model_path = File.join(OUT_DIR, "hmm_gmm_model.marshal")
File.open(model_path, "wb") { |f| f.write(Marshal.dump(model)) }

puts "Done. Model saved to: #{model_path}"
