# bin/cache_manifest_features.rb
# Compute MFCC features for all entries in a manifest and save to cache if missing.

require 'fileutils'
require 'numo/narray'
require_relative '../lib/asr/audio/wav_reader'
require_relative '../lib/asr/preprocess/normalize'
require_relative '../lib/asr/features/mfcc'

PROJ_ROOT = File.expand_path('..', __dir__)
CACHE_DIR = File.join(PROJ_ROOT, 'data', 'cache', 'features')
FileUtils.mkdir_p(CACHE_DIR)

MANIFEST = ARGV[0] || File.join(PROJ_ROOT, 'data', 'manifests', 'test_manifest.tsv')
WAV_ROOT = File.join(PROJ_ROOT, 'data', 'wav', 'test-clean')

def read_manifest(path)
  rows = []
  File.foreach(path) do |line|
    line = line.strip
    next if line.empty?
    parts = line.split("\t")
    next if parts[0].downcase == 'id'
    if parts.length >= 2
      id, wav_path = parts[0], parts[1]
    else
      id = parts[0]
      wav_path = nil
    end
    rows << { id: id, wav: wav_path }
  end
  rows
end

def extract_mfcc_from_wav(wav_path)
  audio = ASR::Audio::WavReader.read(wav_path)
  samples = audio[:samples]
  sr      = audio[:sample_rate]
  samples = ASR::Preprocess::Normalize.peak(samples)
  mfcc_out = ASR::Features::MFCC.extract(samples, sr)
  feats = mfcc_out[:data][:features]
  x = Numo::DFloat.cast(feats)
  x = x.reshape(feats.length, feats.first.length)
  x
end

entries = read_manifest(MANIFEST)
puts "Caching features for #{entries.length} entries from #{MANIFEST}"

entries.each_with_index do |e, idx|
  id = e[:id]
  wav = e[:wav]
  wav ||= File.join(WAV_ROOT, "#{id}.wav")
  cache_path = File.join(CACHE_DIR, "#{id}.marshal")
  if File.exist?(cache_path)
    next
  end
  unless File.exist?(wav)
    # try find by basename
    basename = File.basename(wav)
    matches = Dir.glob(File.join(WAV_ROOT, '**', basename))
    wav = matches.first if matches && !matches.empty?
  end
  unless File.exist?(wav)
    warn "SKIP id=#{id} wav not found"
    next
  end
  begin
    x = extract_mfcc_from_wav(wav)
    File.open(cache_path, 'wb') { |f| f.write(Marshal.dump({ features: x, t: x.shape[0], d: x.shape[1], wav: wav })) }
    if (idx % 100 == 0) || idx == entries.length - 1
      puts "Cached #{idx+1}/#{entries.length} — id=#{id} T=#{x.shape[0]} D=#{x.shape[1]}"
    end
  rescue => ex
    warn "FAILED id=#{id} error=#{ex.class}: #{ex.message}"
  end
end

puts "Done caching."
