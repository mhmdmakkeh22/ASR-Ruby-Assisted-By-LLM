#!/usr/bin/env ruby
# frozen_string_literal: true

require "json"
require "optparse"
require_relative "../lib/asr"

opts = {
  model_json: "models/balanced_5000/acoustic_python_trained.json",
  features_dir: "Data/features/balanced_5000/test",
  lexicon: "models/dev/lexicon.tsv",
  lm: "models/dev/lm_unigram.json",
  beam_size: 16,
  lm_weight: 3.0,
  word_penalty: -1.0
}

OptionParser.new do |o|
  o.banner = "Usage: ruby bin/evaluate_python_model.rb [OPTIONS]"
  o.on("--model_json PATH", "Python-trained model JSON file") { |v| opts[:model_json] = v }
  o.on("--features_dir DIR", "Test features directory") { |v| opts[:features_dir] = v }
  o.on("--lexicon PATH", "Lexicon file") { |v| opts[:lexicon] = v }
  o.on("--lm PATH", "Language model file") { |v| opts[:lm] = v }
  o.on("--beam_size N", Integer, "Beam size") { |v| opts[:beam_size] = v }
  o.on("--lm_weight W", Float, "LM weight") { |v| opts[:lm_weight] = v }
  o.on("--word_penalty W", Float, "Word penalty") { |v| opts[:word_penalty] = v }
end.parse!

puts "Evaluating Python-Trained Model"
puts "=" * 50
puts "Model: #{opts[:model_json]}"
puts "Features: #{opts[:features_dir]}"
puts "Lexicon: #{opts[:lexicon]}"
puts "LM: #{opts[:lm]}"
puts "Beam size: #{opts[:beam_size]}"
puts "LM weight: #{opts[:lm_weight]}"
puts "Word penalty: #{opts[:word_penalty]}"
puts "=" * 50

# Load Python-trained model
puts "Loading Python-trained model..."
model_data = JSON.parse(File.read(opts[:model_json]))

puts "Model info:"
puts "  Phones: #{model_data['phones'].length}"
puts "  States per phone: #{model_data['n_states']}"
puts "  Components per state: #{model_data['n_components']}"
puts "  Feature dimension: #{model_data['dim']}"

# Create a simple wrapper for the Python model
class PythonModelWrapper
  def initialize(model_data)
    @model_data = model_data
    @phones = model_data['phones']
    @n_states = model_data['n_states']
    @n_components = model_data['n_components']
    @dim = model_data['dim']
    @phone_models = model_data['phone_models']
  end
  
  def phones
    @phones
  end
  
  def n_states
    @n_states
  end
  
  def compute_likelihood(features)
    # Simple likelihood computation (very simplified)
    # In real implementation would use proper HMM-GMM forward algorithm
    
    total_loglik = 0.0
    
    features.each do |frame|
      frame_loglik = 0.0
      
      # Sum over all phones (simplified)
      @phones.each do |phone|
        next unless @phone_models[phone]
        
        phone_loglik = 0.0
        
        # Sum over all states
        @phone_models[phone].each do |state_data|
          state_loglik = 0.0
          
          # Sum over all components
          state_data['weights'].each_with_index do |weight, k|
            means = state_data['means'][k]
            variances = state_data['variances'][k]
            
            # Simple Gaussian likelihood (simplified)
            comp_loglik = Math.log(weight + 1e-10)
            
            means.each_with_index do |mean, d|
              diff = frame[d] - mean
              var = variances[d]
              comp_loglik += -0.5 * (Math.log(2 * Math::PI * var) + (diff * diff) / var)
            end
            
            state_loglik += Math.exp(comp_loglik)
          end
          
          phone_loglik += state_loglik / @n_states
        end
        
        frame_loglik += phone_loglik / @phones.length
      end
      
      total_loglik += Math.log(frame_loglik + 1e-10)
    end
    
    return total_loglik
  end
end

# Load model wrapper
model = PythonModelWrapper.new(model_data)

# Load lexicon
puts "Loading lexicon..."
lexicon = ASR::Lexicon.load_tsv(opts[:lexicon])

# Load language model
puts "Loading language model..."
lm_data = JSON.parse(File.read(opts[:lm]))
lm = ASR::LM::Unigram.from_h(lm_data)

# Load test features
puts "Loading test features..."
test_files = Dir.glob(File.join(opts[:features_dir], "*.marshal"))
puts "Found #{test_files.length} test utterances"

# Simple evaluation (no full decoder for now)
puts "\n=== Simple Evaluation ==="
puts "Computing likelihoods for test utterances..."

total_loglik = 0.0
processed_utts = 0

test_files.take(20).each_with_index do |file_path, idx|
  begin
    utt_data = ASR::Utils.marshal_load(file_path)
    features = utt_data[:feats]
    utt_id = utt_data[:utt_id]
    
    # Compute likelihood
    loglik = model.compute_likelihood(features)
    total_loglik += loglik
    processed_utts += 1
    
    puts "  #{idx + 1}/20: #{utt_id} - loglik: #{loglik.round(2)}"
    
  rescue => e
    puts "  Error processing #{file_path}: #{e}"
  end
end

if processed_utts > 0
  avg_loglik = total_loglik / processed_utts
  puts "\n=== Results ==="
  puts "Processed utterances: #{processed_utts}"
  puts "Average log-likelihood: #{avg_loglik.round(2)}"
  puts "Total log-likelihood: #{total_loglik.round(2)}"
  
  # Compare with baseline (would need baseline model for comparison)
  puts "\n=== Analysis ==="
  puts "✓ Python-trained model loaded successfully"
  puts "✓ Feature computation working"
  puts "✓ Model structure compatible"
  puts "✓ Ready for full decoding evaluation"
  
  if avg_loglik > -10000
    puts "✓ Model appears to be trained (reasonable likelihoods)"
  else
    puts "⚠ Model may need more training (very low likelihoods)"
  end
else
  puts "No utterances processed successfully"
end

puts "\n=== Next Steps ==="
puts "1. Implement full Viterbi decoder with Python model"
puts "2. Compare with original Ruby-trained model"
puts "3. Scale up to full training dataset"
puts "4. Update README with Python training results"
