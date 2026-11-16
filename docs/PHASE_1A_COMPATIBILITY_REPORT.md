# VoxVerse — Phase 1A: Environment Compatibility & Audio Foundation Spike Report

> **Status:** Phase 1A Complete  
> **Date:** 2026-11-16  
> **Author:** P1 — Audio and Speaker Profile Engineer (reviewed with P4 — Pipeline Engineer)  
> **Operating Charter:** Adheres strictly to [AGENTS.md](../AGENTS.md)

---

## 1. Actual Environment & Hardware Inspection

An on-metal diagnostic was executed on the host system. The findings are documented below:

| Metric | Measured Value | Analysis & Engineering Implications |
|---|---|---|
| **OS** | Windows 11 Home (Build 10.0.26300, SP0) | 64-bit Windows on ARM |
| **Hardware Architecture** | ARM64 | Qualcomm Snapdragon(R) X — X126100 (Oryon(TM) CPU) |
| **CPU Topography** | 8 physical cores, 8 logical processors | CPU-only environment. No discrete NVIDIA/AMD GPU |
| **Physical Memory** | 15.60 GB total (~10.5 GB available) | Conservative memory limits required for batch audio processing |
| **Disk Space (C:)** | 458.0 GB total, 163.8 GB used, **294.2 GB free** | Headroom is ample for local models and working audio buffers |
| **Python Binary** | `C:\Users\kuchi\AppData\Local\Python\pythoncore-3.14-64\python.exe` | Python 3.14.8 (MSC v.1944 64-bit AMD64) |
| **Python Architecture** | AMD64 (`win_amd64`, 64-bit WindowsPE) | Running under Windows on ARM's x64 emulation layer |
| **Active Virtualenv** | `C:\Users\kuchi\OneDrive\Documents\VOXVERSE\.venv` | Isolated Phase 1A virtual environment |
| **FFmpeg Status** | **Not detected on system PATH** | External binary missing; `winget.exe` is available |

---

## 2. Python & ML Compatibility Comparison

The laptop runs on ARM64 silicon, but the default installed Python interpreter is an x64 AMD64 build running under Windows on ARM emulation. Each core numerical and ML library was evaluated for pre-built wheel availability, clean importation, and actual runtime execution under this configuration.

### Compatibility Matrix

| Component | Tested Version | Interpreter Architecture | Prebuilt Wheel | Import Test | Runtime Compute Smoke Test | Result |
|---|---|---|---|---|---|---|
| **NumPy** | 2.5.5 | Python 3.14 (AMD64) | Yes (`cp314-win_amd64`) | PASSED | Matrix dot-product `(1000, 1000)` float32 BLAS | **VERIFIED WORKING** |
| **SoundFile** | 0.14.0 | Python 3.14 (AMD64) | Yes (bundled libsndfile) | PASSED | Native encode/decode WAV, FLAC, OGG, MP3 | **VERIFIED WORKING** |
| **SciPy** | 1.18.1 | Python 3.14 (AMD64) | Yes (`cp314-win_amd64`) | PASSED | Polyphase signal resampling (48 kHz $\to$ 16 kHz) | **VERIFIED WORKING** |
| **ONNX Runtime** | 1.32.0 | Python 3.14 (AMD64) | Yes (`cp314-win_amd64`) | PASSED | `CPUExecutionProvider` session & tensor inference | **VERIFIED WORKING** |
| **PyTorch CPU** | 2.15.0+cpu | Python 3.14 (AMD64) | Yes (`download.pytorch.org/whl/cpu`) | PASSED | Matmul `(500, 500)` & `nn.Linear` CPU forward pass | **VERIFIED WORKING** |
| **torchaudio CPU**| 2.11.1+cpu | Python 3.14 (AMD64) | Yes (`download.pytorch.org/whl/cpu`) | PASSED | `T.Spectrogram` transform on synthetic 16 kHz tensor | **VERIFIED WORKING** |
| **CTranslate2** | 4.9.0 | Python 3.14 (AMD64) | Yes (`cp314-win_amd64`) | Dry-run verified | Wheel available on PyPI (Candidate for Whisper INT8) | **COMPATIBLE (Wheels exist)** |
| **faster-whisper**| 1.2.1 | Python 3.14 (AMD64) | Yes (`py3-none-any`) | Dry-run verified | Wheel available on PyPI (Candidate for Phase 1B) | **COMPATIBLE (Wheels exist)** |

### Python Environment Verdict

1. **Python 3.14.8 (AMD64 emulation)** is the **sole Python version currently installed** on this host.
2. In Phase 0, there was concern that Python 3.14 was too new for ML packages. However, empirical testing confirmed that **pre-built Windows AMD64 wheels now exist for NumPy 2.5, SoundFile 0.14, SciPy 1.18, ONNX Runtime 1.32, and PyTorch 2.15+cpu**.
3. Emulated x64 execution on Qualcomm Snapdragon X runs smoothly without instruction-set crashes (no missing AVX faults occurred during BLAS or ONNX Runtime execution).
4. **Conclusion:** Creating a separate Python 3.12 or 3.11 environment is **not required** at this stage. The existing `.venv` is stable and sufficient for the audio foundation and VAD layer.

---

## 3. FFmpeg Verification & System Status

- **Command `ffmpeg -version`:** Exited with `CommandNotFoundException`. FFmpeg is not installed on PATH.
- **Command `ffprobe -version`:** Exited with `CommandNotFoundException`.
- **System Package Manager:** `winget.exe` was discovered at `C:\Users\kuchi\AppData\Local\Microsoft\WindowsApps\winget.exe`.
- **Recommended Package:** `Gyan.FFmpeg` (version 9.0.2), maintained by Gyan Doshi, which is the officially endorsed Windows build provider listed on `ffmpeg.org`.
- **Installation Command:**
  ```powershell
  winget install Gyan.FFmpeg
  ```
- **Policy Adherence:** In accordance with Section 8 of the project rules and Section 5 of the Phase 1A specification (*"Ask for explicit user approval before installing system software. Do not download or execute installers silently"*), the system software installation was **not executed silently**.
- **Architectural Graceful Degradation:**
  - `soundfile` (libsndfile 1.2.2+) **natively decodes and encodes WAV, FLAC, OGG, and MP3** without requiring external binaries.
  - Formats requiring container demuxing (M4A/AAC, WebM/Opus) are routed to an FFmpeg subprocess adapter. If FFmpeg is absent, the system raises an explicit `FFmpegNotFoundError` instructing the operator to install FFmpeg.

---

## 4. Audio Processing Architecture (P1 Ownership)

A modular, test-driven audio package was engineered in `voxverse/genai/audio/`:

```
voxverse/genai/audio/
├── __init__.py           # Clean package exports
├── schemas.py            # Pydantic v2 schemas: AudioMetadata, AudioQualityReport, SpeechSegment, VADResult
├── exceptions.py         # Domain exceptions: AudioDecodeError, AudioValidationError, FFmpegNotFoundError, VADInferenceError
├── decoder.py            # AudioDecoder: SoundFile native multi-format + FFmpeg fallback adapter
├── validator.py          # AudioValidator: Measurable RMS, peak amplitude, duration, silence, & clipping checks
├── normalizer.py         # AudioNormalizer: Non-destructive conversion to 16 kHz mono 16-bit PCM WAV
└── vad.py                # SileroVADDetector: Voice Activity Detection via ONNX Runtime CPU
```

### Invariants Enforced
1. **Non-Destructive:** The source input file is never modified in-place (`source == target` raises `AudioProcessingError`).
2. **Deterministic Output:** All outputs target standardized 16,000 Hz, 1 channel (mono), 16-bit linear PCM WAV (`subtype='PCM_16'`).
3. **Polyphase Resampling:** Downsampling/upsampling uses `scipy.signal.resample_poly` for rational anti-aliased filtering.
4. **Channel Downmixing:** Multi-channel audio (e.g. stereo) is downmixed by averaging channels across the channel axis.
5. **No Fabricated Metrics:** RMS and peak amplitude are mathematically computed. Signal-to-noise ratio (SNR) is deliberately not fabricated without a calibrated noise reference.
6. **Resource Safety:** Maximum upload file size is enforced before reading bytes into memory.

---

## 5. Silero VAD Feasibility & CPU Performance

- **Model Acquisition:** Official Silero VAD v5 ONNX model (`silero_vad.onnx`, 2,327,524 bytes, MIT License) acquired from the official repository (`snakers4/silero-vad`).
- **Storage:** Stored locally in `models/silero_vad.onnx` (strictly excluded by `.gitignore`).
- **Runtime:** Loaded via `onnxruntime.InferenceSession` on `CPUExecutionProvider`.
- **Inference Windows:** Sliding window of 512 samples (32 ms at 16,000 Hz) with a `(2, 1, 128)` recurrent state.
- **Empirical Findings:**
  - Digital silence produces speech probability $\approx 0.00059$, yielding **0 false positives**.
  - Synthetic sine tones produce speech probability $\approx 0.00052$, properly identified as non-speech.
  - Per-chunk latency on Qualcomm Oryon CPU: **$\approx 0.25$ ms per 32 ms chunk**.
  - Real-Time Factor (RTF): **$0.0085$ to $0.0094$** (VAD runs **$\approx 115\times$ faster than real-time**).
- **Acoustic Reality Disclosure:** Silero VAD is trained on human speech acoustics. Synthetic tones verify computational stability, latency, and absence of false positive triggers on non-speech. Validation on true human voice recordings is required in Phase 1B to measure Speech False Rejection Rate.

---

## 6. Automated Test Results

Running `pytest tests/ -v`:

```text
============================= test session starts =============================
platform win32 -- Python 3.14.8, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\kuchi\OneDrive\Documents\VOXVERSE
configfile: pyproject.toml
plugins: anyio-4.16.0, asyncio-1.5.0
asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=function, asyncio_default_test_loop_scope=function
collected 88 items

tests\unit\test_audio_decoder.py ..........                              [ 11%]
tests\unit\test_audio_normalizer.py .........                            [ 21%]
tests\unit\test_audio_vad.py .....                                       [ 27%]
tests\unit\test_audio_validator.py .........                             [ 37%]
tests\unit\test_config.py .................                              [ 56%]
tests\unit\test_health.py ........                                       [ 65%]
tests\unit\test_info.py ...........                                      [ 78%]
tests\unit\test_schemas.py ...................                           [100%]

============================= 88 passed in 2.31s ==============================
```

- **Phase 0 Baseline:** All 55 existing tests (config, health, info, API schemas) continue to pass 100%.
- **Phase 1A Audio Suite:** 33 new unit tests covering:
  - Mono and stereo WAV decoding
  - Multi-format decoding (MP3, FLAC, OGG)
  - Non-existent and zero-byte file rejection
  - Corrupted file rejection
  - Max file size enforcement
  - M4A / WebM FFmpeg requirement verification
  - Duration bounds (min duration 0.5s, max duration limits)
  - Digital silence detection
  - Clipping and saturation detection
  - Channel downmixing (stereo $\to$ mono)
  - Resampling across 8 kHz, 22.05 kHz, 44.1 kHz, and 48 kHz
  - In-place mutation prohibition
  - Original file content preservation
  - Silero VAD loading, silence rejection, empty input handling, and RTF tracking

---

## 7. Empirical Benchmarks on Actual Hardware

Executed via `scripts/benchmark_audio_pipeline.py` using deterministic 44.1 kHz stereo audio inputs:

| Audio Duration | Input Size | Decode Latency | Validation Latency | Resample & Normalization | VAD CPU Inference | Total Pipeline Time | VAD RTF | Pipeline RTF | Peak RAM Traced |
|---|---|---|---|---|---|---|---|---|---|
| **5.0 s** | 861.4 KB | 7.96 ms | 4.06 ms | 18.85 ms | 47.14 ms | **78.01 ms** | **0.0094** | **0.0156** | **7.16 MB** |
| **30.0 s** | 5,168.0 KB | 13.44 ms | 19.79 ms | 77.92 ms | 254.48 ms | **365.63 ms** | **0.0085** | **0.0122** | **42.90 MB** |
| **60.0 s** | 10,336.0 KB | 20.29 ms | 41.13 ms | 153.82 ms | 520.04 ms | **735.28 ms** | **0.0087** | **0.0123** | **85.80 MB** |

### Benchmark Observations
1. **Pipeline Speed:** End-to-end audio preparation (decoding 44.1k stereo + validation + polyphase resampling to 16k mono + writing WAV + VAD inference) takes **less than 750 ms for a full 60-second audio track**.
2. **Real-Time Factor:** Total audio pipeline RTF is **$\approx 0.012$** ($\approx 80\times$ faster than real-time).
3. **Memory Footprint:** Peak working RAM is **$\approx 85.8$ MB** for 60 seconds of stereo uncompressed audio, safely within the laptop's 15.6 GB physical capacity.

---

## 8. Remaining Unknowns & Risks

1. **System FFmpeg Missing:** Audio decoding for containerized streams (M4A, AAC, WebM) remains blocked until FFmpeg is installed (`winget install Gyan.FFmpeg`).
2. **ASR Model Quantization under Emulation:** Whisper / faster-whisper CTranslate2 wheels exist, but INT8 CPU quantization performance under x64 ARM64 emulation remains to be benchmarked in Phase 1B.
3. **IndicTrans2 & Fairseq Dependencies:** Translation engine dependencies must be tested for Python 3.14 wheel compatibility when entering Phase 2.

---

## 9. Reproduction Commands

To reproduce every result locally:

```powershell
# 1. Run full test suite
.\.venv\Scripts\pytest.exe tests/ -v

# 2. Run empirical benchmark
.\.venv\Scripts\python.exe scripts/benchmark_audio_pipeline.py

# 3. Verify PyTorch CPU smoke test
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.randn(5, 5).matmul(torch.randn(5, 5)))"

# 4. Verify ONNX Runtime smoke test
.\.venv\Scripts\python.exe -c "import onnxruntime as ort; print(ort.get_available_providers())"
```
