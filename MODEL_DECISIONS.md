# VoxVerse — Model Research & Decisions

> **Status:** Active Engineering — VAD, ASR, and Speaker Representation Models Verified  
> **Last Updated:** 2026-11-17  
> **Authors:** P1 (Audio/Speaker), P2 (Speech/ASR), P3 (TTS/Voice Gen), P4 (Pipeline)

---

## Table of Contents

1. [Overview](#overview)
2. [ASR Candidates](#asr-candidates)
   - [OpenAI Whisper](#openai-whisper)
   - [Faster-Whisper (CTranslate2)](#faster-whisper-ctranslate2)
3. [Translation Candidates](#translation-candidates)
   - [AI4Bharat IndicTrans2](#ai4bharat-indictrans2)
4. [TTS Candidates](#tts-candidates)
   - [AI4Bharat IndicF5](#ai4bharat-indicf5)
   - [Indic-Speak](#indic-speak)
5. [VAD](#vad)
   - [Silero VAD](#silero-vad)
6. [Speaker Embedding Candidates](#speaker-embedding-candidates)
   - [Resemblyzer](#resemblyzer)
   - [SpeechBrain ECAPA-TDNN](#speechbrain-ecapa-tdnn)
7. [Open Research Questions](#open-research-questions)
8. [Verification Checklist](#verification-checklist)

---

## Overview

This document captures the research and evaluation status of every ML model considered for the
VoxVerse pipeline. The pipeline requires models for five tasks:

| Task | Pipeline Stage | Priority | Owner |
|---|---|---|---|
| Voice Activity Detection (VAD) | Pre-ASR | High | P1 (Audio & Speaker Profile Engineer) |
| Automatic Speech Recognition (ASR) | Core | High | P2 (Speech Understanding Engineer) |
| Machine Translation (en → te) | Core | High | P2 (Speech Understanding Engineer) |
| Text-to-Speech (TTS) in Telugu | Core | High | P3 (Voice Generation Engineer) |
| Speaker Conditioning & Voice Preservation | Core | High | P1 (Enrollment) & P3 (Conditioning) |

> [!CAUTION]
> **No model has been verified to work in this environment.** Python 3.14 compatibility, ARM64
> Snapdragon X compatibility, and actual performance numbers are all UNKNOWN until tested. Do NOT
> assume any model will work — every entry below requires hands-on verification.

### Environment Constraints

- **Python:** 3.14.x (very new; limited pre-built wheel availability)
- **Platform:** Windows 11 ARM64 (Snapdragon X X126100), running x64 Python under emulation
- **Hardware:** 8 logical cores (Qualcomm Oryon), ~16 GB RAM, no GPU
- **Inference:** CPU-only, PyTorch or CTranslate2 backend
- **FFmpeg:** Required as system dependency (not yet installed)

---

## ASR Candidates

### OpenAI Whisper

| Field | Details |
|---|---|
| **Model Name** | Whisper |
| **Source** | OpenAI |
| **Purpose** | English speech-to-text (ASR stage) |
| **Repository** | https://github.com/openai/whisper |
| **Documentation** | https://github.com/openai/whisper/blob/main/README.md |
| **License** | MIT |
| **Language Support** | 99 languages including English (primary) and Telugu (for detection) |
| **Telugu Relevance** | Used for English ASR only; Telugu language detection not needed |

#### Model Sizes

| Size | Parameters | English-Only | Multilingual | Relative Speed (GPU) | RAM Estimate (CPU) |
|---|---|---|---|---|---|
| tiny | 39M | `tiny.en` | `tiny` | ~10x | ~1 GB |
| base | 74M | `base.en` | `base` | ~7x | ~1 GB |
| small | 244M | `small.en` | `small` | ~4x | ~2 GB |
| medium | 769M | `medium.en` | `medium` | ~2x | ~5 GB |
| large | 1550M | — | `large-v3` | 1x | ~10 GB |

> [!NOTE]
> RAM estimates are rough approximations for CPU inference. Actual RAM usage on this system
> needs verification. The `.en` (English-only) variants are smaller and faster when only English
> ASR is needed.

#### CPU Performance

- Whisper's official implementation uses PyTorch. CPU inference is **significantly slower** than GPU.
- Expected real-time factor on CPU: **needs verification**. Community reports suggest `base` model
  processes audio at roughly 2-5x slower than real-time on modern CPUs, but this varies widely.
- The `tiny` and `base` models are the most viable for CPU-only use.

#### Compatibility

| Check | Status |
|---|---|
| Python 3.14 compatibility | ✅ **VERIFIED** — `openai-whisper` installed and functioning on Python 3.14 AMD64 |
| ARM64 / Snapdragon X compatibility | ✅ **VERIFIED** — PyTorch CPU FP32 runs cleanly under x64 emulation |
| Windows compatibility | ✅ **VERIFIED** — Runs without FFmpeg when fed NumPy arrays |
| CPU-only inference | ✅ **VERIFIED** — RTF 0.2006 (`tiny.en`), 0.3528 (`base.en`) |

#### Notes

- Requires audio passed as in-memory float32 NumPy arrays to operate without external FFmpeg binary.
- Evaluated on LibriSpeech test-clean: achieves 1.91% WER with `base.en`, but is **2.5x slower** than faster-whisper.

#### Verification Status: ✅ VERIFIED (Evaluation Complete — Secondary Candidate)

---

### Faster-Whisper (CTranslate2)

| Field | Details |
|---|---|
| **Model Name** | Faster-Whisper |
| **Source** | Guillaume Klein (SYSTRAN) |
| **Purpose** | Optimized English speech-to-text (ASR stage) |
| **Repository** | https://github.com/SYSTRAN/faster-whisper |
| **Documentation** | https://github.com/SYSTRAN/faster-whisper/blob/master/README.md |
| **License** | MIT |
| **Language Support** | Same as Whisper (99 languages) |
| **Telugu Relevance** | Used for English ASR only |

#### Key Advantages Over Vanilla Whisper

- Built on **CTranslate2** — an optimized inference engine for Transformer models
- Supports **INT8 quantization** for reduced memory usage and faster CPU inference
- Empirically verified **2.5x faster** on Snapdragon X CPU than vanilla Whisper with identical accuracy
- Lower memory footprint (22.8 MB peak RAM) and compact model weight footprint (76 MB for `base.en` INT8)

#### Model Sizes & Quantization

All Whisper model sizes are available via CTranslate2 conversion. Empirical benchmarks on Snapdragon X CPU:

| Configuration | Quantization | Audio Duration | Latency | RTF | WER (%) | CER (%) | Status |
|---|---|---|---|---|---|---|---|
| `tiny.en` | INT8 | 57.13s | 4.18s | 0.0732 | 2.55% | 1.25% | Alternative (Fast) |
| `tiny.en` | Float32 | 57.13s | 3.58s | 0.0626 | 2.55% | 1.25% | Alternative (Fast) |
| `base.en` | **INT8** | **57.13s** | **8.02s** | **0.1403** | **1.91%** | **0.57%** | **SELECTED PRODUCTION ENGINE** |
| `small.en`| INT8 | 57.13s | 31.58s | 0.5527 | 1.27% | 0.45% | Heavy Alternative |

#### Compatibility

| Check | Status |
|---|---|
| Python 3.14 compatibility | ✅ **VERIFIED** — `ctranslate2 4.9.0` wheel exists and works on Python 3.14 |
| ARM64 / Snapdragon X compatibility | ✅ **VERIFIED** — Runs without instruction faults under Windows on ARM emulation |
| Windows compatibility | ✅ **VERIFIED** — Full Windows AMD64 compatibility confirmed |
| CPU-only inference | ✅ **VERIFIED** — Primary design target, executes 7.13x faster than real-time |

#### Notes

- Decoding audio through P1's `AudioNormalizer` and `soundfile` avoids an upstream PyAV `metadata_errors` bug.
- Selected as the primary production English ASR engine for VoxVerse Phase 1B (see [PHASE_1B_ASR_EVALUATION.md](docs/PHASE_1B_ASR_EVALUATION.md)).

#### Verification Status: ✅ VERIFIED & SELECTED AS PRODUCTION ENGINE

---

## Translation Candidates

### AI4Bharat IndicTrans2 (200M Distilled with Rotary Embeddings)

| Field | Details |
|---|---|
| **Model Name** | IndicTrans2 200M Distilled (Rotary / RoPE) |
| **Source** | AI4Bharat (IIT Madras) / Dr. Raj Dabre |
| **Purpose** | English-to-Telugu text translation |
| **Repository** | https://github.com/AI4Bharat/IndicTrans2 |
| **Paper** | https://arxiv.org/abs/2305.16307 |
| **Hugging Face** | https://huggingface.co/prajdabre/rotary-indictrans2-en-indic-dist-200M |
| **License** | MIT License |
| **Language Support** | 22 Indic languages including **Telugu (tel_Telu)** |
| **Telugu Relevance** | Direct en→te translation (`eng_Latn` -> `tel_Telu`) |

#### Empirical CPU Benchmark on Qualcomm Snapdragon X (FLORES-200 devtest, 50 sentences)

| Configuration | Parameters | Precision | Peak Process RAM | SacreBLEU | chrF | chrF++ | Avg CPU Latency | Status |
|---|---|---|---|---|---|---|---|---|
| **IndicTrans2 200M Distilled** | **211.8 M** | **Float32 CPU** | **1,253.2 MB** | **29.80** | **65.82** | **61.26** | **16.43s / sent** (1.1s for short) | **SELECTED PRODUCTION ENGINE** |

#### Compatibility

| Check | Status | Notes |
|---|---|---|
| Python 3.14 compatibility | ✅ **VERIFIED** | Runs on `transformers 4.57.6` and `sentencepiece 0.2.3` |
| ARM64 / Snapdragon X compatibility | ✅ **VERIFIED** | PyTorch CPU executes without instruction faults under x64 emulation |
| Windows compatibility | ✅ **VERIFIED** | Windows 11 AMD64 emulation confirmed |
| CPU-only inference | ✅ **VERIFIED** | Inference takes ~1.1s for short conversational sentences |

#### Notes & Findings

- Windows MSVC compilation incompatibility with `IndicTransToolkit` was resolved by authoring a pure Python `IndicTransTextNormalizer` utilizing `sacremoses` and `indic-nlp-library`.
- Transformers v5 breaking changes were resolved by pinning `transformers>=4.40,<5.0.0` and using `use_cache=False`.
- High translation naturalness and grammatical correctness verified against official FLORES-200 ground truth (see [PHASE_2_TRANSLATION_ENGINE.md](docs/PHASE_2_TRANSLATION_ENGINE.md)).

#### Verification Status: ✅ VERIFIED & SELECTED AS PRODUCTION ENGINE

---

### Bodhan AI Indic-Translate (Candidate Evaluation)

| Field | Details |
|---|---|
| **Model Name** | Indic-Translate |
| **Source** | Bodhan AI (IIT Madras CoE / MoE) |
| **Purpose** | English-to-Indic translation |
| **Hugging Face** | https://huggingface.co/bodhan-ai/indic-translate |
| **License** | Custom Gated Model License |
| **Parameter Scale** | ~4B effective parameters (~8B footprint) |
| **Hardware Requirement** | >16 GB RAM / High-end GPU |

#### Evaluation Findings

- Evaluated during Phase 2 as an alternative referenced on Hugging Face.
- **Incompatible with local CPU hardware target:** With 15.6 GB total RAM and ~2.3 GB free memory, an 8B LLM model cannot be loaded without fatal out-of-memory aborts or extreme thrashing.
- **Decision:** Rejected for local CPU MVP; documented for future Cloud GPU deployment.

#### Verification Status: ❌ REJECTED FOR LOCAL CPU MVP (Hardware Budget Incompatible)

---

## TTS Candidates

### AI4Bharat IndicF5

| Field | Details |
|---|---|
| **Model Name** | IndicF5-TTS |
| **Source** | AI4Bharat (IIT Madras) |
| **Purpose** | Telugu text-to-speech synthesis with voice cloning |
| **Repository** | https://github.com/AI4Bharat/IndicF5 |
| **Hugging Face** | https://huggingface.co/ai4bharat/IndicF5 (needs verification) |
| **License** | ⚠️ **NEEDS VERIFICATION** — check repository for license |
| **Language Support** | Multiple Indic languages including **Telugu** |
| **Telugu Relevance** | Direct Telugu TTS support |

#### Key Features (As Claimed)

- Based on the F5-TTS architecture (flow-matching based TTS)
- Supports multiple Indic languages
- Voice cloning / speaker adaptation capability (using reference audio)
- Research-stage project

> [!WARNING]
> IndicF5 is a **research project**. It may not be production-ready. API stability, model
> quality, documentation, and ongoing maintenance are all uncertain. Thorough evaluation is
> required before committing to this model.

#### CPU Performance

- F5-TTS is a diffusion/flow-matching model, which typically requires many inference steps.
- CPU inference for diffusion-based TTS models is generally **very slow** (potentially minutes
  per second of audio).
- Actual performance on this hardware: **needs verification**.
- This is potentially the **bottleneck of the entire pipeline** on CPU.

#### Voice Cloning Capability

- Claims to support zero-shot voice cloning using a reference audio sample.
- Quality of voice cloning for Telugu: **needs verification**.
- Speaker embedding compatibility: **needs verification** — it may use its own speaker
  representation rather than external embeddings.

#### Compatibility

| Check | Status |
|---|---|
| Python 3.14 compatibility | ⚠️ **UNKNOWN** |
| ARM64 / Snapdragon X compatibility | ⚠️ **UNKNOWN** |
| Windows compatibility | ⚠️ **UNKNOWN** — research code is typically Linux-first |
| CPU-only inference | ⚠️ **UNKNOWN** — diffusion models are typically GPU-targeted |

#### Notes

- This is the most uncertain component in the pipeline. If IndicF5 is not viable for CPU
  deployment, alternative approaches include:
  - Using a simpler, non-cloning TTS model
  - Falling back to Indic-Speak (see below)
  - Using a generic TTS engine without voice identity preservation
  - Exploring VITS-based models which may be faster on CPU

#### Verification Status: ❌ NOT VERIFIED

---

### Indic-Speak

| Field | Details |
|---|---|
| **Model Name** | Indic-Speak |
| **Source** | AI4Bharat / Community |
| **Purpose** | Alternative Telugu TTS |
| **Repository** | ⚠️ **NEEDS VERIFICATION** — exact repository URL to be confirmed |
| **License** | ⚠️ **NEEDS VERIFICATION** |
| **Language Support** | Multiple Indic languages (Telugu support needs verification) |
| **Telugu Relevance** | Alternative TTS option for Telugu |

#### Status

- Listed as an alternative to IndicF5.
- Less information is currently available about this model.
- Requires research to determine:
  - Architecture (VITS-based? Tacotron? Other?)
  - Telugu language support and quality
  - Voice cloning capabilities
  - CPU inference performance
  - Installation and integration requirements

#### Compatibility

| Check | Status |
|---|---|
| Python 3.14 compatibility | ⚠️ **UNKNOWN** |
| ARM64 / Snapdragon X compatibility | ⚠️ **UNKNOWN** |
| Windows compatibility | ⚠️ **UNKNOWN** |
| CPU-only inference | ⚠️ **UNKNOWN** |

#### Notes

- This model needs significantly more research before it can be evaluated as a candidate.
- It may serve as a fallback if IndicF5 is not viable.

#### Verification Status: ❌ NOT VERIFIED

---

## VAD

### Silero VAD

| Field | Details |
|---|---|
| **Model Name** | Silero VAD |
| **Source** | Silero Team |
| **Purpose** | Voice Activity Detection — identify speech segments in audio |
| **Repository** | https://github.com/snakers4/silero-vad |
| **Documentation** | https://github.com/snakers4/silero-vad/wiki |
| **License** | MIT |
| **Language Support** | Language-agnostic (works on any language) |
| **Telugu Relevance** | Language-independent; used for English input segmentation |

#### Key Features

- Lightweight model (~2 MB)
- Fast CPU inference (claims <1 ms per audio chunk)
- Language-agnostic — detects speech vs. non-speech regardless of language
- PyTorch-based (loaded via `torch.hub` or direct download)
- Well-documented with active maintenance

#### Model Size

| Model | Size | Format |
|---|---|---|
| silero_vad.jit | ~2 MB | TorchScript (JIT) |
| silero_vad.onnx | ~2 MB | ONNX Runtime |

#### CPU Performance

- Designed for real-time CPU inference.
- Per-chunk inference time is typically sub-millisecond on modern CPUs.
- The ONNX variant may offer better performance on some platforms.
- Overall, VAD is expected to be the **fastest stage** in the pipeline.

#### Compatibility

| Check | Status | Notes |
|---|---|---|
| Python 3.14 compatibility | ✅ **VERIFIED** | Runs on `onnxruntime 1.32.0` on Python 3.14.8 |
| ARM64 / Snapdragon X compatibility | ✅ **VERIFIED** | Runs under x64 emulation at ~0.25 ms per 32 ms chunk |
| Windows compatibility | ✅ **VERIFIED** | Windows 11 Build 26300 tested |
| CPU-only inference | ✅ **VERIFIED** | `CPUExecutionProvider` RTF = 0.0085–0.0094 |

#### Notes

- Silero VAD is the de facto standard for open-source voice activity detection.
- Empirically verified in Phase 1A using `silero_vad.onnx` (v5, 2.2 MB) on ONNX Runtime CPU.
- Zero false positives on digital silence and pure synthetic tones.
- Full pipeline integration available in `voxverse.genai.audio.vad.SileroVADDetector`.
- **Phase 1C Architecture Fix:** Silero VAD v5 requires a 64-sample recurrent context prefix prepended to each 512-sample inference chunk (`shape: (1, 576)`). With this context loop implemented in `SileroVADDetector`, human speech detection confidence is >0.999 while non-speech tones remain at 0.000.

#### Verification Status: ✅ VERIFIED (ONNX Runtime CPU with v5 Recurrent Context)

---

## Speaker Embedding Candidates

> [!NOTE]
> Speaker embedding extraction is used to capture the voice identity of the input speaker so
> that the TTS stage can attempt to reproduce a similar voice in the Telugu output. This is an
> **optional enhancement** — the core pipeline can function without it. Voice cloning quality
> is an **open research question**, especially for cross-lingual scenarios (English voice →
> Telugu speech).

### Resemblyzer

| Field | Details |
|---|---|
| **Model Name** | Resemblyzer |
| **Source** | Corentin Jemine |
| **Purpose** | Speaker embedding extraction for voice identity |
| **Repository** | https://github.com/resemble-ai/Resemblyzer |
| **License** | Apache 2.0 |
| **Language Support** | Language-agnostic speaker embeddings |
| **Telugu Relevance** | Evaluated on English input audio to extract speaker identity |

#### Key Features

- Based on the GE2E (Generalized End-to-End) speaker verification model
- Produces 256-dimensional speaker embeddings
- Lightweight and simple API

#### Compatibility

| Check | Status | Notes |
|---|---|---|
| Python 3.14 compatibility | ❌ **INCOMPATIBLE** | `webrtcvad` C extension fails compilation; outdated `typing` dependency corrupts Python 3.14 stdlib. |
| ARM64 / Snapdragon X compatibility | ❌ **BLOCKED** | Cannot install due to compilation errors on Python 3.14. |
| Windows compatibility | ❌ **BLOCKED** | Deprecated build system; abandoned upstream repository. |

#### Verification Status: ❌ INCOMPATIBLE & REJECTED (Python 3.14 Build Incompatibility)

---

### SpeechBrain ECAPA-TDNN

| Field | Details |
|---|---|
| **Model Name** | ECAPA-TDNN (via SpeechBrain) |
| **Source** | SpeechBrain project |
| **Purpose** | Speaker verification and similarity evaluation metrics |
| **Repository** | https://github.com/speechbrain/speechbrain |
| **Hugging Face** | https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb |
| **Documentation** | https://speechbrain.readthedocs.io/ |
| **License** | Apache 2.0 |
| **Language Support** | Language-agnostic speaker embeddings |
| **Telugu Relevance** | Used for cross-lingual speaker similarity evaluation |

#### Key Features

- State-of-the-art speaker verification model
- Produces 192-dimensional speaker embeddings
- Part of SpeechBrain 1.1.2 — pure Python wheel installs cleanly on Python 3.14

#### Compatibility

| Check | Status | Notes |
|---|---|---|
| Python 3.14 compatibility | ✅ **VERIFIED** | `speechbrain 1.1.2` installs and runs cleanly with PyTorch 2.15 CPU |
| ARM64 / Snapdragon X compatibility | ✅ **VERIFIED** | Executes under x64 emulation on Snapdragon X CPU |
| Windows compatibility | ✅ **VERIFIED** | Fully compatible on Windows 11 AMD64 emulation |
| CPU-only inference | ✅ **VERIFIED** | Inference takes ~220 ms per 5s audio chunk |

#### Critical Architectural Finding: Verification vs. TTS Conditioning
During Phase 1C research, a crucial architectural distinction was confirmed:
- SpeechBrain ECAPA-TDNN compresses speech into a fixed **192-dimensional vector** designed for *speaker verification* (same vs. different speaker discrimination).
- Modern zero-shot flow-matching TTS models (e.g., **IndicF5 / F5-TTS**) do **not** take a 192-d or 256-d vector. They condition directly on the **raw reference audio waveform** (canonical 16 kHz WAV) concatenated in the latent acoustic space with reference and target text transcripts.
- Consequently, VoxVerse stores canonical 16 kHz reference audio (`reference.wav`) during enrollment for TTS conditioning, reserving SpeechBrain ECAPA-TDNN solely for **Phase 3 objective voice similarity measurement** (evaluating cosine distance between English input and Telugu output).

#### Verification Status: ✅ VERIFIED COMPATIBLE (Selected for Similarity Evaluation, Not TTS Conditioning)

---

## Open Research Questions

The following questions do not have answers yet and represent genuine technical unknowns that
must be investigated:

### 1. Cross-Lingual Voice Cloning Quality

**Question:** How well can a Telugu TTS model reproduce the voice characteristics of an English
speaker using speaker embeddings extracted from English audio?

- This is fundamentally challenging because voice characteristics (pitch, timbre, speaking rate)
  interact with language-specific phonetics.
- Quality may range from "somewhat similar" to "completely different" depending on the TTS
  model's architecture and training data.
- **Status:** Open research question. No claims should be made about quality until tested.

### 2. CPU Inference Feasibility for TTS

**Question:** Is diffusion/flow-matching based TTS (IndicF5) practical on CPU?

- These models typically require many inference steps (50-100+ denoising steps).
- On GPU, this takes seconds. On CPU, it could take **minutes per second of audio**.
- If CPU TTS is too slow, alternatives include:
  - VITS-based models (single forward pass, much faster)
  - Reducing denoising steps (quality trade-off)
  - Accepting long processing times for offline use
- **Status:** Needs benchmarking on target hardware.

### 3. Total Pipeline Memory Budget

**Question:** Can all pipeline models fit in 16 GB RAM simultaneously?

- Rough estimates (unverified):
  - Silero VAD: ~50 MB
  - Faster-Whisper base (INT8): ~150 MB
  - IndicTrans2 200M: ~800 MB
  - TTS model: ~1-4 GB (unknown)
  - Speaker embedding: ~100-500 MB
  - Python + libraries overhead: ~500 MB
  - **Total estimate: ~2-6 GB** (highly uncertain)
- The 16 GB RAM should be sufficient, but if the TTS model is large, lazy loading (load/unload
  models per stage) may be necessary.
- **Status:** Needs measurement with actual models loaded.

### 4. Python 3.14 Ecosystem Readiness

**Question:** Do all required libraries have working wheels for Python 3.14?

- Python 3.14 was released in 2025. As of November 2026, most major libraries should have
  support, but ML ecosystem libraries (PyTorch, CTranslate2, etc.) sometimes lag.
- If any critical library lacks Python 3.14 support, options include:
  - Building from source
  - Using Python 3.12 or 3.13 instead
  - Finding alternative libraries
- **Status:** Must be verified during environment setup.

### 5. ARM64 Emulation Overhead

**Question:** What is the performance overhead of running x64 Python under emulation on
Snapdragon X?

- Windows on ARM includes an x64 emulation layer. The performance overhead is typically
  10-30% for compute-heavy tasks, but can vary.
- CTranslate2 and PyTorch may have ARM64-native builds that could avoid emulation entirely.
- **Status:** Needs benchmarking.

---

## Verification Checklist

Before committing to any model, the following verification steps must be completed. Each step
should be documented with results (pass/fail, measurements, notes).

### Environment Verification

- [ ] **FFmpeg Installation:** Install FFmpeg on Windows ARM64 and verify it works
- [ ] **PyTorch Installation:** Install PyTorch (CPU) on Python 3.14 and verify `import torch` works
- [ ] **ONNX Runtime Installation:** Install onnxruntime on Python 3.14 and verify import works
- [ ] **CTranslate2 Installation:** Install ctranslate2 on Python 3.14 and verify import works
- [ ] **RAM Baseline:** Measure Python process memory before loading any models

### Per-Model Verification

#### Silero VAD
- [ ] Install and import silero-vad successfully
- [ ] Load the VAD model (JIT or ONNX)
- [ ] Run VAD on a test audio file (10 seconds of speech)
- [ ] Measure inference time and memory usage
- [ ] Verify segmentation output is reasonable

#### Faster-Whisper (Primary ASR Candidate)
- [ ] Install faster-whisper and ctranslate2 successfully
- [ ] Download the `base` model (or `tiny` for initial testing)
- [ ] Transcribe a 10-second English audio clip
- [ ] Measure transcription accuracy (compare to known transcript)
- [ ] Measure inference time (real-time factor)
- [ ] Measure memory usage with model loaded
- [ ] Test INT8 quantization vs. float32

#### OpenAI Whisper (Fallback ASR)
- [ ] Install openai-whisper successfully (only if Faster-Whisper fails)
- [ ] Download the `base` model
- [ ] Transcribe the same test audio clip
- [ ] Compare accuracy and speed to Faster-Whisper

#### IndicTrans2
- [ ] Clone the IndicTrans2 repository
- [ ] Install all dependencies (transformers, sentencepiece, etc.)
- [ ] Download the 200M distilled model
- [ ] Translate 10 test sentences from English to Telugu
- [ ] Have a Telugu speaker evaluate translation quality
- [ ] Measure inference time per sentence
- [ ] Measure memory usage with model loaded
- [ ] Test the 1.1B model if RAM allows, and compare quality

#### IndicF5
- [ ] Clone the IndicF5 repository
- [ ] Install all dependencies
- [ ] Download the Telugu TTS model
- [ ] Synthesize audio from 3 Telugu sentences
- [ ] Evaluate audio quality (intelligibility, naturalness)
- [ ] Measure inference time per sentence
- [ ] Measure memory usage with model loaded
- [ ] Test voice cloning with an English reference audio clip
- [ ] Evaluate cloned voice similarity

#### Indic-Speak (Fallback TTS)
- [ ] Locate the correct repository and documentation
- [ ] Determine if Telugu is supported
- [ ] Repeat the same TTS evaluation steps as IndicF5

#### Resemblyzer (Primary Speaker Embedding)
- [ ] Install resemblyzer successfully
- [ ] Extract embedding from a test audio clip
- [ ] Verify embedding shape and values are reasonable
- [ ] Measure inference time and memory usage
- [ ] Determine if TTS model can accept these embeddings

#### SpeechBrain ECAPA-TDNN (Fallback Speaker Embedding)
- [ ] Install speechbrain successfully
- [ ] Load the ECAPA-TDNN model
- [ ] Extract embedding from the same test audio clip
- [ ] Compare embedding quality to Resemblyzer
- [ ] Measure inference time and memory usage

### Integration Verification

- [ ] **Memory Budget:** Load all selected models simultaneously and measure total RAM usage
- [ ] **End-to-End Pipeline:** Run a complete pipeline (audio → text → translation → speech)
  with all models loaded
- [ ] **End-to-End Timing:** Measure total pipeline time for a 30-second audio clip
- [ ] **Concurrent Requests:** Test with 2 simultaneous requests — verify stability and memory

### Test Audio Requirements

For consistent verification, prepare the following test audio files:

1. **Short clip:** 5-10 seconds of clear English speech (single speaker, no background noise)
2. **Medium clip:** 30 seconds of English speech (single speaker)
3. **Noisy clip:** 10 seconds of English speech with background noise
4. **Multi-speaker:** 15 seconds with two English speakers
5. **Silence-heavy:** 30 seconds with significant pauses between speech segments

### Documentation Requirements

For each model that passes verification:

- [ ] Document the exact installation command that worked
- [ ] Document the exact model version/checkpoint used
- [ ] Document measured performance numbers (time, memory)
- [ ] Document any workarounds or patches needed
- [ ] Document any configuration tuning performed
