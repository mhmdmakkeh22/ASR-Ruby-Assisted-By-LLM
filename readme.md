# ASR-Ruby-Assisted-By-LLM

## What This Project Is

This repository contains a complete **Automatic Speech Recognition (ASR)** system built in Ruby (with Python acceleration helpers), developed with the assistance of an LLM (Large Language Model). The system transcribes spoken audio into text using a classical **HMM-GMM** (Hidden Markov Model + Gaussian Mixture Model) pipeline trained on the [LibriSpeech](https://www.openslr.org/12) dataset.

The project has three distinct implementations that show the evolution of the system:

| Directory | What it is |
|-----------|-----------|
| `ASR-LLM-Word-Model/` | First prototype — word-level recognition |
| `ASR-LLM-Sentence-Model/` | Improved prototype — sentence-level recognition |
| `ASR-Modular/` | Final, clean, modular implementation (primary codebase) |
| `ASR-Complete/` | Production scripts + pre-trained model files |

---

## High-Level Architecture

```
  WAV Audio
     │
     ▼
  Pre-processing       (normalize, pre-emphasis, framing, windowing)
     │
     ▼
  Feature Extraction   (MFCC — 13 coefficients per 10ms frame)
     │
     ▼
  Acoustic Model       (HMM-GMM per phoneme, trained with Baum-Welch EM)
     │
     ▼
  Beam Search Decoder  (Viterbi + unigram language model)
     │
     ▼
  Transcribed Text
```

---

## Full Repository File Map

```
ASR-Ruby-Assisted-By-LLM/
│
├── README.md                          ← this file
│
├── ASR-LLM-Word-Model/                ← Prototype 1: word-level ASR
│   ├── Gemfile / Gemfile.lock         ← Ruby dependencies
│   ├── bin/                           ← executable scripts
│   ├── lib/asr/                       ← core library (same structure as Sentence-Model)
│   └── tools/                         ← data preparation utilities
│
├── ASR-LLM-Sentence-Model/            ← Prototype 2: sentence-level ASR
│   ├── Gemfile / Gemfile.lock         ← Ruby dependencies
│   ├── bin/                           ← executable scripts
│   ├── lib/asr/                       ← core library modules
│   ├── tools/                         ← data preparation utilities
│   ├── decoder_beam_results.json      ← saved beam-search decode output
│   ├── decoder_greedy_results.json    ← saved greedy decode output
│   └── nn_decoder_results.json        ← saved NN-decoder output
│
├── ASR-Modular/                       ← PRIMARY CODEBASE (final, clean version)
│   ├── GemFile / GemFile.lock         ← Ruby dependencies
│   ├── Lib/                           ← core library
│   ├── Bin/                           ← all runnable scripts
│   ├── test/                          ← unit tests
│   ├── tmp/                           ← temporary test artifacts
│   ├── train_full_28k.py              ← fast Python trainer
│   ├── train_fast.py                  ← fast Python trainer (variant)
│   ├── train_python_full.py           ← full dataset Python trainer
│   ├── train_python_full_dataset.py   ← full dataset Python trainer (variant)
│   ├── train_python_to_ruby.py        ← exports Python-trained model to Ruby format
│   ├── evaluate_python_complete.py    ← Python-side evaluation (WER)
│   ├── extract_features_full_dataset.py ← batch MFCC feature extraction
│   ├── train_full_dataset.rb          ← Ruby full-dataset training launcher
│   └── eval_full_dataset.rb           ← Ruby full-dataset evaluation launcher
│
└── ASR-Complete/                      ← Production deliverable + saved models
    ├── README.md / README_FINAL.md / README_SCIENTIFIC.md
    ├── docs/
    │   ├── PERFORMANCE_ANALYSIS.md
    │   └── TECHNICAL_DETAILS.md
    ├── models/                        ← pre-trained model files (binary + JSON)
    │   ├── lexicon.tsv                ← word → phoneme mapping
    │   ├── lm_unigram.json            ← unigram language model
    │   └── full_dataset/             ← multiple trained acoustic models + results
    └── scripts/                       ← training and evaluation scripts
```

---

## File-by-File Explanation

### `ASR-Modular/GemFile`
Ruby dependency file (like `package.json` for Node). Declares that the project requires Ruby >= 3.1.0 and the `minitest` gem for unit testing. Run `bundle install` to install dependencies.

---

### `ASR-Modular/Lib/asr.rb`
**Entry point for the library.** Loads all submodules in the correct order:
`utils → audio → preprocess → features → lexicon → lm → am → am_optimized → decoder → eval`
Any script that writes `require_relative "../Lib/asr"` gets the entire system.

---

### `ASR-Modular/Lib/asr/utils.rb`
**Shared helper functions** used across all modules. Contains:
- `ASR::Utils.marshal_load/save` — serialize/deserialize Ruby objects to disk
- `ASR::Utils.tokenize(text)` — splits a sentence into uppercase words
- `ASR::Utils::EPS` — numerical epsilon constant (prevents log(0) errors)
- JSON I/O helpers

---

### `ASR-Modular/Lib/asr/audio.rb`
**WAV file reader.** Reads a `.wav` audio file and returns a flat array of PCM audio samples (16-bit signed integers converted to floats in the range [-1, 1]). Handles mono 16 kHz audio as expected by LibriSpeech.

---

### `ASR-Modular/Lib/asr/preprocess.rb`
**Audio preprocessing pipeline.** Applied to the raw waveform before feature extraction:
1. **Normalize** — scale samples so the loudest sample = 1.0
2. **Pre-emphasis** — apply a high-pass filter (`y[t] = x[t] - 0.97*x[t-1]`) to boost high frequencies
3. **Framing** — cut the audio into overlapping 25ms windows every 10ms
4. **Windowing** — multiply each frame by a Hamming window to reduce edge effects

---

### `ASR-Modular/Lib/asr/features.rb`
**MFCC feature extraction.** Converts each audio frame into a 13-dimensional Mel-Frequency Cepstral Coefficient (MFCC) vector. Steps:
1. FFT (Fast Fourier Transform) → power spectrum
2. Mel filterbank (26 triangular filters on the Mel scale)
3. Log compression
4. DCT (Discrete Cosine Transform) → 13 coefficients
5. CMVN (Cepstral Mean and Variance Normalization) per utterance

Defines three preset configurations: `DEFAULT_MFCC`, `HIGH_RES_MFCC`, `FAST_MFCC`.

---

### `ASR-Modular/Lib/asr/lexicon.rb`
**Pronunciation lexicon.** Loads a `.tsv` file mapping every word to its sequence of phonemes (e.g., `HELLO → HH AH L OW`). Provides:
- `Lexicon.load_tsv(path)` — parse the TSV file
- `phones_for(word)` — look up a word's phoneme sequence
- A simple grapheme-to-phoneme (G2P) fallback for unknown words

---

### `ASR-Modular/Lib/asr/lm.rb`
**Unigram language model.** Estimates the probability P(word) for every word in the training vocabulary using add-one (Laplace) smoothing. Serializes to/from JSON so Python and Ruby can share the same LM file.

---

### `ASR-Modular/Lib/asr/am.rb`
**Acoustic model — HMM-GMM.** The core machine learning component. Contains two classes:
- **`DiagGMM`** — a Gaussian Mixture Model with diagonal covariance. Given an MFCC frame, returns log P(frame | phone state).
- **`HMMGMM`** — one HMM per phone. Each HMM has N states (default 3–5), each state modeled by a `DiagGMM`. Trained with the **Baum-Welch** (forward-backward) Expectation-Maximization algorithm.

---

### `ASR-Modular/Lib/asr/am_optimized.rb`
**Faster variant of the acoustic model.** Uses Viterbi-style hard-assignment EM instead of full Baum-Welch. Trades a small amount of accuracy for significantly faster training on large datasets.

---

### `ASR-Modular/Lib/asr/decoder.rb`
**Beam search decoder.** Converts a sequence of MFCC frames into a word sequence. Uses:
- **Viterbi-style beam search** — keeps only the top N hypotheses at each time step
- **Language model rescoring** — multiplies acoustic scores by the unigram LM weight
- **Word insertion penalty** — controls how often the decoder inserts word boundaries
- **Vocabulary filtering** — silently drops words whose phonemes are not in the acoustic model

Key parameters: `beam_size` (200), `beam_delta` (20 dB pruning), `lm_weight` (0.8), `word_penalty` (-0.2).

---

### `ASR-Modular/Lib/asr/eval.rb`
**Evaluation metrics.** Implements **Word Error Rate (WER)**:
`WER = (substitutions + deletions + insertions) / total reference words`
Uses dynamic programming (edit distance) to align hypothesis and reference transcripts.

---

### `ASR-Modular/Lib/asr/convert_librispeech_flac_to_wav.rb`
Helper utility embedded in the library to convert LibriSpeech `.flac` files to `.wav` by shelling out to `ffmpeg` or `sox`.

---

## ASR-Modular/Bin/ — Runnable Scripts

All scripts accept `--help` for full usage. They all load the library via `require_relative "../Lib/asr"`.

### Data Preparation

| Script | What it does |
|--------|-------------|
| `make_manifest.rb` | Scans a folder of WAV files + transcriptions and writes a JSONL manifest (one line per utterance: id, text, audio path) |
| `librispeech_make_manifest.rb` | Same but handles LibriSpeech's nested folder structure automatically |
| `create_train_clean_100_manifest.rb` | Builds the manifest specifically for LibriSpeech train-clean-100 |
| `make_toy_dataset.rb` | Creates a tiny 10-utterance dataset for quick smoke-testing |
| `convert_librispeech_flac_to_wav.rb` | Batch-converts all `.flac` files in a directory to `.wav` |
| `copy_trans.rb` | Copies transcription files alongside audio files to match expected directory layout |
| `create_balanced_dataset.rb` | Samples an equal number of utterances per speaker to balance the training set |
| `create_train_subset.rb` | Creates a smaller random subset of the training manifest for faster experiments |
| `analyze_dataset_balance.rb` | Prints speaker/duration statistics to check dataset balance |

### Feature Extraction

| Script | What it does |
|--------|-------------|
| `extract_features.rb` | Reads WAV files from a manifest, computes MFCCs, saves one `.marshal` file per utterance |
| `extract_features_enhanced.rb` | Same but adds delta and delta-delta coefficients (39-dim features instead of 13) |

### Model Building

| Script | What it does |
|--------|-------------|
| `build_lexicon.rb` | Reads a manifest and builds a `lexicon.tsv` from the CMU Pronouncing Dictionary |
| `build_lm.rb` | Reads transcriptions and builds a `lm_unigram.json` language model |

### Training

| Script | What it does |
|--------|-------------|
| `train_am.rb` | **Main training script.** Trains an HMM-GMM acoustic model. Args: `--features_dir`, `--lexicon`, `--out`, `--n_states`, `--n_components`, `--n_iter` |
| `train_am_optimized.rb` | Same but uses the faster Viterbi EM trainer |
| `train_simple.rb` | Stripped-down training with hardcoded defaults — useful for a quick first run |
| `train_balanced.rb` | Runs training on a balanced dataset |
| `train_full_dev_clean.rb` | Trains on the full LibriSpeech dev-clean split |
| `train_and_eval.rb` | Chains training immediately followed by evaluation in one script |
| `quick_train_dev_clean.rb` | Fast one-command training on dev-clean with sensible defaults |
| `benchmark_training.rb` | Measures training throughput (utterances/second) for different configurations |

### Decoding & Evaluation

| Script | What it does |
|--------|-------------|
| `decode.rb` | Loads a trained acoustic model and decodes feature files. Args: `--features_dir`, `--lexicon`, `--lm`, `--am`, `--out` |
| `evaluate_test.rb` | Runs decoding on the test set and reports WER |
| `eval_WER.rb` | Standalone WER calculator: given a hypotheses JSON and references JSON, prints overall WER |
| `evaluate_python_model.rb` | Loads a Python-trained model and evaluates it with the Ruby decoder |
| `inspect_model.rb` | Prints model metadata: number of phones, states, parameters, etc. |

---

## ASR-Modular/test/ — Unit Tests

| File | What it tests |
|------|--------------|
| `test_audio.rb` | Reads and writes a short WAV file; checks sample values round-trip correctly |
| `test_mfcc_cmvn.rb` | Runs the full MFCC pipeline on a synthetic signal; checks output shape and CMVN normalization |
| `test_gmm_em.rb` | Creates a `DiagGMM`, runs one EM iteration on random data; checks weights sum to 1 and variances stay positive |
| `test_decoder_smoke.rb` | Builds a tiny 3-phone model and decodes a 5-frame feature sequence; checks output is a list of strings |

Run all tests with: `bundle exec ruby -Ilib test/test_audio.rb`

---

## ASR-Modular/ — Python Scripts

These bypass the slow Ruby EM training for large datasets. They output a Ruby-compatible JSON so the Ruby decoder can load the result.

| Script | What it does |
|--------|-------------|
| `train_full_28k.py` | **Primary fast trainer.** Trains on all 28,539 LibriSpeech utterances using mini-batch gradient descent. Outputs `.marshal` + `.json` model files. |
| `train_fast.py` | Variant with different hyperparameters |
| `train_python_full.py` | Full-dataset trainer using a more complete GMM update rule |
| `train_python_full_dataset.py` | Another variant used during development |
| `train_python_to_ruby.py` | Converts a Python `pickle` model to Ruby `.marshal` format |
| `evaluate_python_complete.py` | Decodes the test/dev set with a Python beam-search decoder and reports WER |
| `extract_features_full_dataset.py` | Batch MFCC extraction using NumPy — much faster than Ruby for large datasets |
| `train_full_dataset.rb` | Ruby wrapper that calls the Python trainer and handles file paths |
| `eval_full_dataset.rb` | Ruby wrapper for the evaluation step |

---

## ASR-LLM-Sentence-Model/ and ASR-LLM-Word-Model/

Earlier prototypes. They use Numo::NArray and split the library into deeper subdirectories:

```
lib/asr/
├── audio/wav_reader.rb       ← WAV reader
├── features/
│   ├── mfcc.rb               ← MFCC orchestrator
│   ├── fft.rb                ← FFT implementation
│   ├── mel_filterbank.rb     ← Mel filter bank
│   ├── power_spectrum.rb     ← |FFT|² computation
│   ├── dct.rb                ← Discrete Cosine Transform
│   └── cmvn.rb               ← Cepstral Mean & Variance Normalization
├── math/
│   ├── gmm.rb                ← GMM class (uses Numo::NArray)
│   ├── hmm_gmm.rb            ← HMM-GMM with Baum-Welch training
│   └── logsumexp.rb          ← numerically stable log-sum-exp
└── preprocess/
    ├── framing.rb            ← frame the signal
    ├── windowing.rb          ← Hamming window
    ├── normalize.rb          ← amplitude normalization
    └── pre_emphasis.rb       ← high-pass filter
```

### bin/ scripts

| Script | What it does |
|--------|-------------|
| `run_pipeline.rb` | **End-to-end pipeline** — runs all steps in order: build lexicon → build LM → cache features → train → decode → report WER |
| `train_hmm_gmm.rb` | Trains the HMM-GMM model from a manifest file |
| `decoder_beam.rb` | Decodes using beam search; writes results to `decoder_beam_results.json` |
| `decoder_greedy_words.rb` | Simpler greedy decoder (picks the single best phone at each frame) |
| `nn_sentence_decoder.rb` | Experimental neural-network-based decoder |
| `nn_decoder_leaveone_eval.rb` | Leave-one-out cross-validation for the NN decoder |
| `eval_hmm_gmm.rb` | Evaluates the HMM-GMM model and prints WER |
| `build_manifest_dev.rb` | Builds a manifest for the LibriSpeech dev-clean split |
| `cache_manifest_features.rb` | Pre-computes and caches MFCC features for all utterances in a manifest |
| `compute_ll_report.rb` | Computes per-utterance log-likelihood and writes a report |

### tools/ scripts

| Script | What it does |
|--------|-------------|
| `build_lexicon.rb` | Builds word → phoneme mapping from training transcriptions |
| `build_unigram_lm.rb` | Counts word frequencies and writes a unigram LM JSON file |
| `build_word_gmms.rb` | Trains a separate per-word GMM (direct word model, not phone-based) |
| `build_word_gmms_worker.rb` | Worker process for parallel word-GMM training |
| `forced_align.rb` | Aligns a known transcription to an audio file using Viterbi decoding |
| `make_test_manifest.rb` | Builds a manifest for the test split |

---

## ASR-Complete/scripts/ — Production Training Scripts

Final, tuned versions of the Python training scripts used to train the released models:

| Script | What it does |
|--------|-------------|
| `train_full_28k.py` | Trains on all 28,539 utterances — the main production training script |
| `train_working.py` | Earlier working version |
| `train_fixed.py` | Bug-fixed version of the trainer |
| `train_better.py` | Improved version with better hyperparameters |
| `train_optimal.py` | Hyperparameter-tuned version |
| `train_results_focused.py` | Variant focused on maximizing WER improvement |
| `train_ruby_fast.py` | Produces a model compatible with the Ruby decoder |
| `train_ultimate.py` | Final combined training approach |
| `train_ultimate_ruby.py` | Same but exports to Ruby format |
| `evaluate_fast.py` | Quick evaluation script for development |
| `evaluate_python_complete.py` | Full evaluation with per-utterance WER breakdown |
| `extract_features_full_dataset.py` | Batch MFCC extraction for the full training set |
| `create_train_lexicon_lm.py` | Builds both the lexicon and LM in one step |
| `test_ruby_model.py` | Sanity-checks a Ruby-format model by loading and scoring a few frames |
| `train_full_dataset.rb` | Ruby wrapper that calls the Python trainer |
| `eval_full_dataset.rb` | Ruby wrapper for the evaluation step |

---

## ASR-Complete/models/ — Saved Model Files

All model files are pre-trained and ready to use for decoding.

| File | What it is |
|------|-----------|
| `lexicon.tsv` | Evaluation-set lexicon: word + phoneme sequence per row |
| `train_lexicon.tsv` | Training-set lexicon (larger vocabulary) |
| `lm_unigram.json` | Unigram language model (word → log-probability) |
| `train_lm.json` | Language model built from training transcriptions |
| `full_dataset/acoustic_full_28k.marshal` | Best acoustic model trained on 28,539 utterances (Ruby binary format) |
| `full_dataset/acoustic_full_28k.json` | Same model in JSON (human-readable, Python-compatible) |
| `full_dataset/acoustic_full_28k_metrics.json` | WER and log-likelihood scores for this model |
| `full_dataset/acoustic_full_28k_evaluation_results.json` | Per-utterance decode results |
| `full_dataset/acoustic_full_28k_hypotheses.json` | Decoder text output for all test utterances |
| `full_dataset/acoustic_*.marshal/.json` | Other model checkpoints from different training runs |
| `full_dataset/acoustic_*_results.json` | Evaluation results for each checkpoint |

---

## ASR-Complete/docs/

| File | What it contains |
|------|-----------------|
| `PERFORMANCE_ANALYSIS.md` | Benchmarks comparing Ruby vs. Python training speed, memory usage, and WER across model configurations |
| `TECHNICAL_DETAILS.md` | Deep-dive into MFCC parameters, HMM topology, beam search tuning, and dataset statistics |

---

## End-to-End Flow

```
1. Convert audio       Bin/convert_librispeech_flac_to_wav.rb
2. Build manifest      Bin/librispeech_make_manifest.rb  →  manifest.jsonl
3. Extract features    Bin/extract_features.rb           →  data/features/<id>.marshal
4. Build lexicon       Bin/build_lexicon.rb              →  models/lexicon.tsv
5. Build LM            Bin/build_lm.rb                   →  models/lm_unigram.json
6. Train model         Bin/train_am.rb  (Ruby)           →  models/acoustic.marshal
                  OR   train_full_28k.py  (Python)       →  models/acoustic.json
7. Decode              Bin/decode.rb                     →  exp/hyp.txt
8. Evaluate            Bin/eval_WER.rb                   →  WER printed to console
```

`ruby Bin/train_and_eval.rb` runs steps 6–8 automatically.
`ruby bin/run_pipeline.rb` (in ASR-LLM-Sentence-Model) runs all 8 steps in one shot.

---

## Technology Summary

| Component | Language | Key technique |
|-----------|----------|--------------|
| Feature extraction | Ruby / Python | MFCC with CMVN |
| Acoustic model | Ruby / Python | HMM-GMM, Baum-Welch EM |
| Language model | Ruby | Unigram with Laplace smoothing |
| Decoder | Ruby | Viterbi beam search |
| Fast training | Python (NumPy) | Online mini-batch gradient descent |
| Evaluation | Ruby | Edit-distance WER |
| Dataset | — | LibriSpeech train-clean-100 (28,539 utterances) |
