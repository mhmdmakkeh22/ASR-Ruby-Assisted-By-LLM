#!/usr/bin/env ruby
# frozen_string_literal: true

require_relative "../lib/asr"

model_path = ARGV[0] || "models/dev/acoustic_full_optimized.marshal"

puts "Model Inspection: #{model_path}"
puts "=" * 50

begin
  model = ASR::Utils.marshal_load(model_path)
  
  puts "Model Overview:"
  puts "  Phones: #{model.phones.keys.length}"
  puts "  States per phone: #{model.n_states}"
  puts "  Components per state: #{model.n_components}"
  puts "  Feature dimension: #{model.dim}"
  puts "  Model class: #{model.class.name}"
  
  puts "\nPhone List (first 20):"
  model.phones.keys.take(20).each_with_index do |phone, i|
    puts "  #{(i+1).to_s.rjust(2)}. #{phone}"
  end
  
  # Check a sample phone model
  sample_phone = model.phones.keys.first
  phone_model = model.phones[sample_phone]
  
  puts "\nSample Phone Details: '#{sample_phone}'"
  puts "  States: #{phone_model.gmms.length}"
  puts "  GMM 0 components: #{phone_model.gmms[0].w.length}"
  puts "  GMM 0 weights: #{phone_model.gmms[0].w.map { |w| w.round(4) }.join(', ')}"
  puts "  GMM 0 mean (first 3): #{phone_model.gmms[0].mu[0].take(3).map { |x| x.round(3) }.join(', ')}"
  puts "  GMM 0 var (first 3): #{phone_model.gmms[0].var[0].take(3).map { |x| x.round(3) }.join(', ')}"
  
  # Check if SIL phone exists (critical for decoding)
  if model.has_phone?("SIL")
    puts "\n✓ SIL phone found"
    sil_model = model.phones["SIL"]
    puts "  SIL states: #{sil_model.gmms.length}"
  else
    puts "\n❌ SIL phone NOT found (this will cause decoding issues!)"
  end
  
  # Check transition probabilities
  puts "\nTransition Matrix (first phone):"
  sample_phone_model = model.phones[model.phones.keys.first]
  puts "  Self-transition probs: #{sample_phone_model.self_probs.map { |p| p.round(4) }.join(', ')}"
  
rescue => e
  puts "Error loading model: #{e.message}"
  puts "Backtrace:"
  puts e.backtrace.first(5)
end
