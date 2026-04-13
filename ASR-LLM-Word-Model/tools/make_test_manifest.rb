# tools/make_test_manifest.rb
# Create a manifest TSV for test-clean by scanning dataset/test-clean
require 'fileutils'
PROJ_ROOT = File.expand_path('..', __dir__)
WAV_ROOT = File.join(PROJ_ROOT, 'data', 'wav', 'test-clean')
OUT = File.join(PROJ_ROOT, 'data', 'manifests', 'test_manifest.tsv')
FileUtils.mkdir_p(File.dirname(OUT))

files = Dir.glob(File.join(WAV_ROOT, '**', '*.{flac,wav}'), File::FNM_CASEFOLD)
File.open(OUT, 'w') do |f|
  f.puts "id\tpath\ttext"
  files.each do |p|
    id = File.basename(p, File.extname(p))
    f.puts "#{id}\t#{p}\t"
  end
end
puts "Wrote #{OUT} (#{files.length} entries)"
