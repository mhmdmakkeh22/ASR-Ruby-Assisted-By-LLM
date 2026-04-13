#!/usr/bin/env python3
"""
Create Lexicon and Language Model from FULL train-clean-100 data
Proper resources for better ASR performance
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import math

def create_lexicon_from_train(manifest_file, output_lexicon_file):
    """Create comprehensive lexicon from training data"""
    print(f"Creating lexicon from {manifest_file}")
    
    word_phone_map = {}
    word_counts = Counter()
    
    # Load training manifest
    with open(manifest_file, 'r') as f:
        for line_num, line in enumerate(f):
            if line_num % 1000 == 0:
                print(f"  Processing line {line_num}")
            
            try:
                utterance = json.loads(line.strip())
                words = utterance.get('words', [])
                
                for word in words:
                    word_counts[word] += 1
                    
                    # Create phonetic representation (simplified)
                    if word not in word_phone_map:
                        # Simple phone mapping based on word structure
                        phones = word_to_phones(word)
                        word_phone_map[word] = phones
                        
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    # Filter words by frequency (keep words that appear at least 2 times)
    filtered_words = {word: phones for word, phones in word_phone_map.items() 
                     if word_counts[word] >= 2}
    
    # Sort by frequency
    sorted_words = sorted(filtered_words.items(), 
                        key=lambda x: word_counts[x[0]], reverse=True)
    
    # Write lexicon
    with open(output_lexicon_file, 'w') as f:
        for word, phones in sorted_words:
            f.write(f"{word}\t{' '.join(phones)}\n")
    
    print(f"Lexicon created: {len(sorted_words)} words")
    print(f"Output: {output_lexicon_file}")
    
    return word_phone_map, word_counts

def word_to_phones(word):
    """Convert word to phonetic representation"""
    # Simplified phonetic rules
    phone_map = {
        # Vowels
        'a': ['AH'], 'e': ['IH'], 'i': ['IY'], 'o': ['AO'], 'u': ['UW'],
        'y': ['IY'],
        # Consonants
        'b': ['B'], 'c': ['K'], 'd': ['D'], 'f': ['F'], 'g': ['G'],
        'h': ['HH'], 'j': ['JH'], 'k': ['K'], 'l': ['L'], 'm': ['M'],
        'n': ['N'], 'p': ['P'], 'q': ['K'], 'r': ['R'], 's': ['S'],
        't': ['T'], 'v': ['V'], 'w': ['W'], 'x': ['K', 'S'], 'y': ['Y'],
        'z': ['Z'],
        # Common combinations
        'ch': ['CH'], 'sh': ['SH'], 'th': ['TH'], 'ng': ['NG', 'G'],
        'ph': ['F'], 'wh': ['W', 'HH']
    }
    
    # Convert to lowercase and map
    word_lower = word.lower()
    phones = []
    
    # Handle common patterns
    if word_lower.startswith('th'):
        phones.extend(['TH'])
        word_lower = word_lower[2:]
    elif word_lower.startswith('sh'):
        phones.extend(['SH'])
        word_lower = word_lower[2:]
    elif word_lower.startswith('ch'):
        phones.extend(['CH'])
        word_lower = word_lower[2:]
    
    # Map remaining letters
    for char in word_lower:
        if char in phone_map:
            phones.extend(phone_map[char])
    
    # Ensure at least some phones
    if not phones:
        phones = ['SIL', 'AH', 'SIL']
    
    # Add silence at beginning and end
    phones = ['SIL'] + phones + ['SIL']
    
    return phones

def create_language_model(manifest_file, output_lm_file, max_vocab=10000):
    """Create unigram language model from training data"""
    print(f"Creating language model from {manifest_file}")
    
    word_counts = Counter()
    total_words = 0
    
    # Count words in training data
    with open(manifest_file, 'r') as f:
        for line_num, line in enumerate(f):
            if line_num % 1000 == 0:
                print(f"  Processing line {line_num}")
            
            try:
                utterance = json.loads(line.strip())
                words = utterance.get('words', [])
                
                for word in words:
                    word_counts[word] += 1
                    total_words += 1
                    
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    # Filter vocabulary
    vocab_words = word_counts.most_common(max_vocab)
    
    # Calculate probabilities
    word_probs = {}
    for word, count in vocab_words:
        # Add-one smoothing
        prob = (count + 1) / (total_words + len(vocab_words))
        word_probs[word] = math.log(prob)
    
    # Unknown word probability
    unknown_prob = 1.0 / (total_words + len(vocab_words) + 1)
    word_probs['<UNK>'] = math.log(unknown_prob)
    
    # Create language model structure
    lm_data = {
        'vocab_size': len(vocab_words),
        'total_words': total_words,
        'word_probs': word_probs,
        'unknown_word_log_prob': math.log(unknown_prob),
        'smoothing': 'add_one',
        'created_from': manifest_file
    }
    
    # Write language model
    with open(output_lm_file, 'w') as f:
        json.dump(lm_data, f, indent=2)
    
    print(f"Language model created: {len(vocab_words)} words")
    print(f"Total words in training: {total_words}")
    print(f"Output: {output_lm_file}")
    
    return lm_data

def create_enhanced_lexicon(manifest_file, output_lexicon_file):
    """Create enhanced lexicon with multiple pronunciations"""
    print(f"Creating enhanced lexicon from {manifest_file}")
    
    word_pronunciations = {}
    word_counts = Counter()
    
    # Load training data
    with open(manifest_file, 'r') as f:
        for line_num, line in enumerate(f):
            if line_num % 1000 == 0:
                print(f"  Processing line {line_num}")
            
            try:
                utterance = json.loads(line.strip())
                words = utterance.get('words', [])
                
                for word in words:
                    word_counts[word] += 1
                    
                    # Create multiple pronunciations for common words
                    if word not in word_pronunciations:
                        pronunciations = generate_pronunciations(word)
                        word_pronunciations[word] = pronunciations
                        
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    # Filter by frequency
    filtered_pronunciations = {word: prons for word, prons in word_pronunciations.items() 
                             if word_counts[word] >= 2}
    
    # Sort by frequency
    sorted_words = sorted(filtered_pronunciations.items(), 
                        key=lambda x: word_counts[x[0]], reverse=True)
    
    # Write enhanced lexicon
    with open(output_lexicon_file, 'w') as f:
        for word, pronunciations in sorted_words:
            for pron in pronunciations:
                f.write(f"{word}\t{' '.join(pron)}\n")
    
    print(f"Enhanced lexicon created: {len(sorted_words)} words")
    print(f"Total pronunciations: {sum(len(prons) for prons in filtered_pronunciations.values())}")
    print(f"Output: {output_lexicon_file}")
    
    return filtered_pronunciations, word_counts

def generate_pronunciations(word):
    """Generate multiple pronunciations for a word"""
    base_phones = word_to_phones(word)
    
    # Remove SIL from base for variations
    if base_phones[0] == 'SIL':
        base_phones = base_phones[1:-1]
    elif base_phones[-1] == 'SIL':
        base_phones = base_phones[:-1]
    
    pronunciations = [base_phones]
    
    # Add variations for common words
    word_lower = word.lower()
    
    # Variation 1: Reduced vowels
    if len(word) > 3:
        reduced_phones = []
        for phone in base_phones:
            if phone in ['AH', 'IH', 'IY', 'AE', 'EY', 'AO', 'OW', 'UW']:
                reduced_phones.append('AH')  # Reduce to schwa
            else:
                reduced_phones.append(phone)
        pronunciations.append(reduced_phones)
    
    # Variation 2: Final consonant devoicing
    if word_lower.endswith(('s', 'z')):
        final_phones = base_phones[:-1] + ['S']
        pronunciations.append(final_phones)
    elif word_lower.endswith('d'):
        final_phones = base_phones[:-1] + ['T']
        pronunciations.append(final_phones)
    
    # Variation 3: T-flapping
    if 't' in word_lower and len(word) > 2:
        flapped_phones = []
        for i, phone in enumerate(base_phones):
            if phone == 'T' and i > 0 and i < len(base_phones) - 1:
                flapped_phones.append('D')  # Flapped T
            else:
                flapped_phones.append(phone)
        pronunciations.append(flapped_phones)
    
    # Remove duplicates
    unique_pronunciations = []
    seen = set()
    for pron in pronunciations:
        pron_tuple = tuple(pron)
        if pron_tuple not in seen:
            seen.add(pron_tuple)
            unique_pronunciations.append(pron)
    
    return unique_pronunciations

def main():
    parser = argparse.ArgumentParser(description='Create Lexicon and LM from Training Data')
    parser.add_argument('--train_manifest', required=True, help='Training manifest file')
    parser.add_argument('--output_lexicon', required=True, help='Output lexicon file')
    parser.add_argument('--output_lm', required=True, help='Output language model file')
    parser.add_argument('--enhanced', action='store_true', help='Create enhanced lexicon')
    parser.add_argument('--max_vocab', type=int, default=10000, help='Maximum vocabulary size')
    
    args = parser.parse_args()
    
    print("🚀 Creating Lexicon and Language Model from Training Data")
    print("=" * 60)
    print(f"Training manifest: {args.train_manifest}")
    print(f"Output lexicon: {args.output_lexicon}")
    print(f"Output language model: {args.output_lm}")
    print(f"Enhanced lexicon: {args.enhanced}")
    print(f"Max vocabulary: {args.max_vocab}")
    print("=" * 60)
    
    # Check if manifest exists
    if not os.path.exists(args.train_manifest):
        print(f"Error: Training manifest {args.train_manifest} not found!")
        return
    
    # Create lexicon
    if args.enhanced:
        print(f"\n=== Creating Enhanced Lexicon ===")
        lexicon_data, word_counts = create_enhanced_lexicon(args.train_manifest, args.output_lexicon)
    else:
        print(f"\n=== Creating Basic Lexicon ===")
        lexicon_data, word_counts = create_lexicon_from_train(args.train_manifest, args.output_lexicon)
    
    # Create language model
    print(f"\n=== Creating Language Model ===")
    lm_data = create_language_model(args.train_manifest, args.output_lm, args.max_vocab)
    
    # Summary
    print(f"\n=== Creation Summary ===")
    print(f"Lexicon created with {len(lexicon_data)} words")
    print(f"Language model created with {lm_data['vocab_size']} words")
    print(f"Total training words: {lm_data['total_words']}")
    
    # Show sample words
    print(f"\n=== Sample Words ===")
    sample_words = list(lexicon_data.keys())[:10]
    for word in sample_words:
        pronunciations = lexicon_data[word]
        print(f"  {word}: {pronunciations[0] if pronunciations else 'N/A'}")
    
    print(f"\n✅ Lexicon saved to: {args.output_lexicon}")
    print(f"✅ Language model saved to: {args.output_lm}")
    print(f"\n🎯 Ready for improved ASR training!")

if __name__ == "__main__":
    main()
