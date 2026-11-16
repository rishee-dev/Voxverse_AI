# VoxVerse — Architecture Document

> **Status:** Phase 0 — Foundation  
> **Last Updated:** 2026-11-16  
> **Authors:** P1, P2, P3, P4

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Responsibilities & Ownership](#component-responsibilities--ownership)
3. [Data Flow](#data-flow)
4. [Module Dependency Graph](#module-dependency-graph)
5. [API Design](#api-design)
6. [Storage Design](#storage-design)
7. [Configuration Management](#configuration-management)
8. [Error Handling Strategy](#error-handling-strategy)
9. [Security Considerations](#security-considerations)
10. [Future Considerations](#future-considerations)
11. [Technology Stack](#technology-stack)

---

## System Overview

VoxVerse is an AI-powered voice translation platform that converts spoken English audio into
spoken Telugu audio. The system operates as a multi-stage pipeline where each stage is responsible
for a discrete transformation of the input data.

The platform is designed for **CPU-only execution** on consumer hardware (targeting ARM64/Snapdragon X
and x64 systems) with no GPU dependency. All model inference runs locally; no cloud AI APIs are used
for the core pipeline.

### High-Level Architecture

```
┌─────────────┐    ┌────────────┐    ┌───────┐    ┌───────┐    ┌─────────────┐    ┌───────┐    ┌────────────┐
│ Audio Input  │───▶│ Validation │───▶│  VAD  │───▶│  ASR  │───▶│ Translation │───▶│  TTS  │───▶│ Audio Out  │
│ (Upload/Mic) │    │ & Preproc  │    │       │    │       │    │  (en → te)  │    │       │    │ (WAV/MP3)  │
└─────────────┘    └────────────┘    └───────┘    └───────┘    └─────────────┘    └───────┘    └────────────┘
```

### Design Principles

- **Modularity:** Each pipeline stage is an independent module with a defined interface. Stages can
  be swapped, upgraded, or tested in isolation.
- **Fail-Fast:** Invalid inputs are rejected at the earliest possible stage. Each stage validates
  its own inputs and outputs.
- **Observability:** Every pipeline execution produces structured logs and timing metrics.
- **Data Privacy:** Voice data is treated as sensitive PII. No data leaves the local machine.
  Temporary files are cleaned up after processing.
- **CPU-First:** All model selections prioritize CPU inference performance. Quantized and optimized
  model variants are preferred.

---

## Component Responsibilities & Ownership

### P1 — Audio and Speaker Profile Engineer

| Component | Responsibility |
|---|---|
| `voxverse/genai/audio/` | Audio I/O, format validation, resampling (16 kHz), normalization |
| `voxverse/genai/audio/` | Voice Activity Detection (Silero VAD) — speech segmentation and gap filtering |
| `voxverse/genai/speaker/` | Voice enrollment, authorized speaker reference management |
| `voxverse/genai/speaker/` | Speaker profile storage, consent verification, and deletion lifecycle |
| Quality validation | Audio SNR, clipping detection, and input recording quality checks |

### P2 — Speech Understanding Engineer

| Component | Responsibility |
|---|---|
| `voxverse/genai/asr/` | Speech-to-text (ASR) via Whisper / faster-whisper CPU INT8 |
| `voxverse/genai/asr/` | English language validation and per-segment confidence scoring |
| `voxverse/genai/translation/` | English-to-Telugu text translation using AI4Bharat IndicTrans2 |
| `voxverse/genai/translation/` | Text post-processing, Indian language script normalization, validation |
| Benchmarking | CPU execution latency, memory footprint, and ASR/MT quality tracking |

### P3 — Voice Generation Engineer

| Component | Responsibility |
|---|---|
| `voxverse/genai/tts/` | Telugu speech synthesis (AI4Bharat IndicF5 / Indic-Speak research candidates) |
| `voxverse/genai/tts/` | Speaker conditioning and voice characteristic preservation |
| Alternative TTS | CPU-compatible fallback TTS research and adapter implementations |
| `voxverse/genai/evaluation/` | Speaker similarity evaluation (embedding cosine distance + subjective scoring) |
| Audio QA | Generated Telugu audio quality assessment and output formatting |

### P4 — Pipeline Engineer

| Component | Responsibility |
|---|---|
| `voxverse/api/` | FastAPI application factory, routing, middleware, error handling |
| `voxverse/api/dependencies/` | Dependency injection (settings, logging, future services) |
| `voxverse/genai/pipeline/` | End-to-end pipeline orchestration and batch job processing |
| Pipeline telemetry | Job metrics, stage timings, error tracking, and audit trail |
| `voxverse/storage/` | Local storage management and SQLite development metadata |

### Shared Ownership

| Component | Responsibility | Owners |
|---|---|---|
| `voxverse/config/` | Centralized settings, environment management (.env) | P4 (primary), All |
| `voxverse/config/logging.py` | Structured JSON/text logging and request correlation | P4 (primary), All |
| `tests/` | Unit and integration test suites for owned components | Each owner |
| Documentation | Architecture, model decisions, API specs | All |

---

## Data Flow

### Detailed Pipeline Stages

#### Stage 1: Audio Input

- **Input:** Audio file upload (WAV, MP3, FLAC, OGG, M4A) or raw PCM stream
- **Output:** Raw audio bytes + metadata (format, sample rate, duration, channels)
- **Owner:** P1

#### Stage 2: Validation & Preprocessing

- **Input:** Raw audio bytes + metadata
- **Processing:**
  - Validate file format and integrity (FFmpeg probe)
  - Check duration limits (configurable, default max 300 seconds)
  - Check file size limits (configurable, default max 50 MB)
  - Resample to 16 kHz mono (standard for speech models)
  - Normalize audio levels
- **Output:** Preprocessed audio tensor (16 kHz, mono, float32)
- **Owner:** P1

#### Stage 3: Voice Activity Detection (VAD)

- **Input:** Preprocessed audio tensor
- **Processing:**
  - Detect speech vs. non-speech segments using Silero VAD
  - Merge adjacent segments within configurable gap threshold
  - Filter segments below minimum duration threshold
- **Output:** List of speech segments with timestamps `[(start_ms, end_ms), ...]`
- **Owner:** P1

#### Stage 4: Automatic Speech Recognition (ASR)

- **Input:** Preprocessed audio tensor + speech segments
- **Processing:**
  - Run Whisper/Faster-Whisper on each speech segment
  - Produce per-segment transcriptions with confidence scores
  - Detect language to confirm English input
- **Output:** Transcription result — list of `TranscriptionSegment(text, start, end, confidence, language)`
- **Owner:** P2

#### Stage 5: Translation (English → Telugu)

- **Input:** English text segments
- **Processing:**
  - Tokenize English text for IndicTrans2
  - Run translation model inference
  - Post-process Telugu text (normalization, punctuation)
- **Output:** Telugu text segments with alignment to source
- **Owner:** P2

#### Stage 6: Text-to-Speech (TTS) & Speaker Conditioning

- **Input:** Telugu text segments + speaker reference representation (from authorized profile)
- **Processing:**
  - Synthesize Telugu speech conditioned on speaker characteristics
  - Preserve voice identity using speaker conditioning (IndicF5 / alternative TTS)
  - Generate audio at target sample rate
- **Output:** Synthesized audio tensor(s) per segment
- **Owner:** P3

#### Stage 7: Output Assembly & Metrics Recording

- **Input:** Synthesized audio segments + original timing information
- **Processing:**
  - Concatenate audio segments with appropriate silence gaps
  - Apply crossfade between segments
  - Encode to output format (WAV default)
  - Record stage execution metrics, memory usage, and job telemetry
- **Output:** Final translated playable audio file + processing metrics
- **Owner:** P4 (Pipeline orchestration) & P1 (Audio export)

### Data Flow Diagram (Detailed)

```
User Request (HTTP POST /api/v1/translate)
│
├─▶ [Request Validation] ── invalid ──▶ HTTP 400/422
│
├─▶ [Audio Extraction] ── corrupt file ──▶ HTTP 400
│         │
│         ▼
│   [Preprocessing]
│   (resample, normalize)
│         │
│         ▼
│   [VAD: Silero]
│   (speech segments)
│         │
│         ▼
│   [ASR: Whisper]
│   (English text + timestamps)
│         │
│         ▼
│   [Translation: IndicTrans2]
│   (Telugu text)
│         │
│         ▼
│   [TTS: IndicF5 / Indic-Speak]
│   (Telugu audio segments)
│         │
│         ▼
│   [Output Assembly]
│   (concatenate, encode)
│         │
│         ▼
│   [Response: translated audio file]
│
├─▶ [Cleanup: delete temp files]
│
└─▶ [Logging: pipeline metrics]
```

---

## Module Dependency Graph

The dependency graph describes which modules depend on which other modules. Dependencies flow
downward; modules at the bottom are depended upon by modules above them.

```
voxverse.api
├── voxverse.pipeline
│   ├── voxverse.audio
│   │   ├── voxverse.audio.vad        (depends on: silero-vad)
│   │   ├── voxverse.audio.io         (depends on: ffmpeg, soundfile, librosa)
│   │   └── voxverse.audio.preprocess (depends on: numpy, torch)
│   ├── voxverse.asr
│   │   └── voxverse.asr.whisper_engine (depends on: faster-whisper OR whisper)
│   ├── voxverse.translation
│   │   └── voxverse.translation.indictrans (depends on: indicTrans2, sentencepiece)
│   └── voxverse.tts
│       ├── voxverse.tts.indicf5      (depends on: indicf5 — TBD)
│       └── voxverse.speaker          (depends on: resemblyzer OR speechbrain)
├── voxverse.config                   (depends on: pydantic-settings)
└── voxverse.utils                    (depends on: standard library, structlog)
```

### External Dependency Categories

| Category | Libraries | Notes |
|---|---|---|
| **Web Framework** | FastAPI, Uvicorn, python-multipart | API layer |
| **Audio Processing** | librosa, soundfile, pydub | Audio I/O and manipulation |
| **ML Runtime** | PyTorch (CPU), CTranslate2 | Model inference |
| **ASR** | faster-whisper or openai-whisper | Speech recognition |
| **Translation** | IndicTrans2, sentencepiece | English-to-Telugu translation |
| **TTS** | IndicF5 / Indic-Speak (TBD) | Telugu speech synthesis |
| **VAD** | silero-vad | Voice activity detection |
| **Speaker** | resemblyzer or speechbrain | Speaker embedding |
| **Config** | pydantic, pydantic-settings, python-dotenv | Configuration management |
| **Logging** | structlog | Structured logging |
| **Testing** | pytest, pytest-asyncio, httpx | Test framework |
| **System** | FFmpeg (external binary) | Audio format conversion |

---

## API Design

### Base URL

```
http://localhost:8000
```

### API Versioning

All pipeline endpoints are versioned under `/api/v1/`. Health and system endpoints are unversioned.

### Endpoints

#### System Endpoints

| Method | Path | Description | Status |
|---|---|---|---|
| `GET` | `/health` | Health check — returns service status | **Implemented** |
| `GET` | `/api/v1/info` | System info — models loaded, versions, capabilities | **Implemented** |

#### Pipeline Endpoints (Planned)

| Method | Path | Description | Status |
|---|---|---|---|
| `POST` | `/api/v1/translate` | Full pipeline: upload audio → receive translated audio | Planned (Phase 2+) |
| `POST` | `/api/v1/transcribe` | ASR only: upload audio → receive English transcription | Planned (Phase 2+) |
| `POST` | `/api/v1/translate-text` | Translation only: English text → Telugu text | Planned (Phase 2+) |
| `POST` | `/api/v1/synthesize` | TTS only: Telugu text → Telugu audio | Planned (Phase 2+) |
| `GET` | `/api/v1/jobs/{job_id}` | Poll job status for async processing | Planned (Phase 3+) |
| `GET` | `/api/v1/jobs/{job_id}/result` | Download result of completed job | Planned (Phase 3+) |

#### Request/Response Formats

**POST /api/v1/translate** (Planned)

```
Request:
  Content-Type: multipart/form-data
  Fields:
    - audio: file (required) — input audio file
    - source_language: string (default: "en") — source language code
    - target_language: string (default: "te") — target language code
    - output_format: string (default: "wav") — output audio format
    - preserve_voice: boolean (default: false) — attempt voice identity preservation

Response (synchronous, small files):
  Content-Type: audio/wav
  Headers:
    X-Pipeline-Duration-Ms: <total_ms>
    X-Transcription: <english_text> (URL-encoded)
    X-Translation: <telugu_text> (URL-encoded)

Response (async, large files):
  Content-Type: application/json
  {
    "job_id": "uuid",
    "status": "processing",
    "estimated_duration_seconds": 30
  }
```

**GET /health**

```
Response:
  Content-Type: application/json
  {
    "status": "healthy",
    "timestamp": "2026-11-16T10:30:00Z",
    "version": "0.1.0"
  }
```

**GET /api/v1/info**

```
Response:
  Content-Type: application/json
  {
    "project": "VoxVerse",
    "version": "0.1.0",
    "pipeline_stages": {
      "vad": {"model": "silero-vad", "status": "not_loaded"},
      "asr": {"model": "faster-whisper-base", "status": "not_loaded"},
      "translation": {"model": "indictrans2-en-te", "status": "not_loaded"},
      "tts": {"model": "indicf5", "status": "not_loaded"}
    },
    "capabilities": {
      "source_languages": ["en"],
      "target_languages": ["te"],
      "audio_formats": ["wav", "mp3", "flac", "ogg", "m4a"],
      "max_duration_seconds": 300,
      "max_file_size_mb": 50
    }
  }
```

---

## Storage Design

### Filesystem Layout

```
VOXVERSE/
├── voxverse/                   # Main application package
│   ├── __init__.py
│   ├── api/                    # FastAPI application (P1)
│   │   ├── __init__.py
│   │   ├── app.py              # FastAPI app factory
│   │   ├── routes/             # Route definitions
│   │   └── middleware/         # CORS, logging, error handling
│   ├── audio/                  # Audio processing (P1)
│   │   ├── __init__.py
│   │   ├── io.py               # Audio file I/O
│   │   ├── preprocess.py       # Resampling, normalization
│   │   └── vad.py              # Voice activity detection
│   ├── asr/                    # Speech recognition (P2)
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract ASR interface
│   │   └── whisper_engine.py   # Whisper implementation
│   ├── translation/            # Text translation (P3)
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract translation interface
│   │   └── indictrans.py       # IndicTrans2 implementation
│   ├── tts/                    # Text-to-speech (P4)
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract TTS interface
│   │   └── indicf5.py          # IndicF5 implementation
│   ├── speaker/                # Speaker embeddings (P4)
│   │   ├── __init__.py
│   │   └── embedder.py         # Speaker embedding extraction
│   ├── pipeline/               # Pipeline orchestration (P1)
│   │   ├── __init__.py
│   │   ├── executor.py         # Pipeline execution engine
│   │   └── models.py           # Pipeline data models
│   ├── config/                 # Configuration (P1)
│   │   ├── __init__.py
│   │   └── settings.py         # Pydantic settings
│   └── utils/                  # Shared utilities (All)
│       ├── __init__.py
│       ├── logging.py          # Structured logging setup
│       └── timing.py           # Performance timing utilities
├── tests/                      # Test suite
│   ├── conftest.py             # Shared fixtures
│   ├── test_api/               # API tests (P1)
│   ├── test_audio/             # Audio processing tests (P1)
│   ├── test_asr/               # ASR tests (P2)
│   ├── test_translation/       # Translation tests (P3)
│   ├── test_tts/               # TTS tests (P4)
│   └── test_pipeline/          # Integration tests (P1)
├── models/                     # Downloaded model files (git-ignored)
│   ├── whisper/
│   ├── indictrans2/
│   ├── indicf5/
│   └── silero/
├── data/                       # Runtime data (git-ignored)
│   ├── uploads/                # Temporary uploaded files
│   ├── outputs/                # Generated output files
│   └── cache/                  # Processing cache
├── docs/                       # Additional documentation
├── scripts/                    # Utility scripts
│   └── download_models.py      # Model download helper
├── .env.example                # Environment variable template
├── .gitignore
├── pyproject.toml              # Project metadata and build config
├── requirements.txt            # Pinned dependencies
├── ARCHITECTURE.md             # This file
├── MODEL_DECISIONS.md          # Model research and decisions
└── README.md                   # Project README
```

### Model Storage

Models are stored under `models/` and are **not committed to Git** (listed in `.gitignore`).
Each model directory contains:
- The model files (weights, config, tokenizer)
- A `manifest.json` with model version, download date, and checksum

```json
{
  "model_name": "faster-whisper-base",
  "version": "1.0.0",
  "downloaded_at": "2026-11-16T10:00:00Z",
  "source_url": "https://huggingface.co/...",
  "sha256": "abc123...",
  "size_bytes": 150000000
}
```

### Temporary File Handling

- Upload files are stored in `data/uploads/` with UUID filenames
- Output files are stored in `data/outputs/` with UUID filenames
- All temporary files are cleaned up after response delivery or after a configurable TTL
  (default: 1 hour)
- The cleanup process runs as a background task in the FastAPI application

---

## Configuration Management

### Approach

Configuration is managed through **environment variables** with a layered override system:

1. **Defaults:** Defined in `voxverse/config/settings.py` using Pydantic `BaseSettings`
2. **`.env` file:** Local overrides loaded from `.env` (not committed to Git)
3. **Environment variables:** System-level overrides (highest priority)

### Configuration Schema

```python
class VoxVerseSettings(BaseSettings):
    """Application configuration with environment variable support."""

    # --- Application ---
    app_name: str = "VoxVerse"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR

    # --- Server ---
    host: str = "127.0.0.1"
    port: int = 8000

    # --- Audio Constraints ---
    max_upload_size_mb: int = 50
    max_audio_duration_seconds: int = 300
    target_sample_rate: int = 16000

    # --- Model Paths ---
    models_dir: str = "models"
    whisper_model_size: str = "base"  # tiny, base, small, medium, large
    indictrans_model_path: str = "models/indictrans2"
    tts_model_path: str = "models/indicf5"
    vad_model_path: str = "models/silero"

    # --- Pipeline ---
    vad_threshold: float = 0.5
    vad_min_speech_duration_ms: int = 250
    vad_max_gap_ms: int = 500
    asr_beam_size: int = 5
    asr_language: str = "en"

    # --- Storage ---
    upload_dir: str = "data/uploads"
    output_dir: str = "data/outputs"
    temp_file_ttl_seconds: int = 3600

    model_config = SettingsConfigDict(
        env_prefix="VOXVERSE_",
        env_file=".env",
        env_file_encoding="utf-8",
    )
```

### Environment Variable Examples

```bash
# .env.example
VOXVERSE_DEBUG=false
VOXVERSE_LOG_LEVEL=INFO
VOXVERSE_HOST=127.0.0.1
VOXVERSE_PORT=8000
VOXVERSE_MAX_UPLOAD_SIZE_MB=50
VOXVERSE_MAX_AUDIO_DURATION_SECONDS=300
VOXVERSE_WHISPER_MODEL_SIZE=base
VOXVERSE_MODELS_DIR=models
```

---

## Error Handling Strategy

### Principles

1. **Typed Errors:** Each pipeline stage defines its own exception types inheriting from a
   common `VoxVerseError` base.
2. **Error Propagation:** Errors propagate up through the pipeline with full context (stage name,
   input metadata, timing, partial results).
3. **User-Facing Messages:** Internal errors are mapped to user-friendly HTTP error responses.
   Internal details (stack traces, model internals) are never exposed to clients.
4. **Structured Logging:** All errors are logged with structured context for debugging.

### Exception Hierarchy

```
VoxVerseError (base)
├── AudioError
│   ├── AudioFormatError          — unsupported or corrupt audio format
│   ├── AudioDurationError        — audio exceeds max duration
│   ├── AudioSizeError            — file exceeds max size
│   └── AudioProcessingError      — resampling/normalization failure
├── VADError                      — voice activity detection failure
├── ASRError
│   ├── ASRModelLoadError         — failed to load ASR model
│   ├── ASRTranscriptionError     — transcription failed
│   └── ASRLanguageError          — detected language is not English
├── TranslationError
│   ├── TranslationModelLoadError — failed to load translation model
│   └── TranslationInferenceError — translation inference failed
├── TTSError
│   ├── TTSModelLoadError         — failed to load TTS model
│   └── TTSSynthesisError         — speech synthesis failed
├── PipelineError                 — pipeline orchestration failure
└── ConfigurationError            — invalid configuration
```

### HTTP Error Mapping

| Exception | HTTP Status | User Message |
|---|---|---|
| `AudioFormatError` | 400 | "Unsupported audio format. Supported: WAV, MP3, FLAC, OGG, M4A." |
| `AudioDurationError` | 400 | "Audio duration exceeds the maximum of {max} seconds." |
| `AudioSizeError` | 413 | "File size exceeds the maximum of {max} MB." |
| `ASRLanguageError` | 400 | "Input audio does not appear to be in English." |
| `*ModelLoadError` | 503 | "Service temporarily unavailable. A required model is not loaded." |
| `*InferenceError` | 500 | "An internal error occurred during processing. Please try again." |
| `PipelineError` | 500 | "An internal error occurred. Please try again." |

---

## Security Considerations

### Voice Data as PII

Voice recordings are biometric data and personally identifiable information (PII). VoxVerse treats
all audio data with the following protections:

1. **Local Processing Only:** All inference runs locally. No audio data is transmitted to external
   services or APIs.
2. **No Persistent Storage of Voice Data:** Uploaded audio and generated output are stored only
   temporarily for the duration of processing. Files are deleted after response delivery or after
   the configured TTL (default: 1 hour).
3. **No Logging of Audio Content:** Audio file contents are never written to log files. Only
   metadata (duration, format, size) is logged.
4. **Secure Temporary Files:** Temporary files use UUID-based names and are stored in a directory
   with restricted permissions.

### Consent Requirements

> [!IMPORTANT]
> Any deployment of VoxVerse that processes voice data from third parties MUST implement consent
> collection. The current development version does not include a consent mechanism — this is a
> requirement for any production deployment.

- Users must be informed that their voice is being recorded and processed.
- Users must consent to the translation and (if enabled) voice identity preservation.
- A mechanism for data deletion requests must be available.

### Data Deletion

- `DELETE /api/v1/jobs/{job_id}` (Planned) — delete all data associated with a job
- Background cleanup task purges expired temporary files
- No analytics or telemetry data is collected

### Network Security

- The API server binds to `127.0.0.1` (localhost only) by default.
- No authentication is implemented in Phase 0. **Authentication is required before any network
  exposure.**
- CORS is configured to allow only localhost origins in development.

### Model Security

- Models are downloaded from known sources (Hugging Face, GitHub releases).
- Model checksums should be verified after download (planned for `download_models.py`).
- Model files are not committed to version control.

---

## Future Considerations

### GPU Acceleration

- The current architecture is CPU-only. GPU support can be added by:
  - Configuring PyTorch with CUDA/ROCm backend
  - Using CTranslate2 with CUDA for Faster-Whisper
  - Enabling GPU inference in IndicTrans2 and TTS models
- The abstract base classes for each stage allow GPU-optimized implementations to be swapped in
  without changing the pipeline orchestration.

### Streaming Pipeline

- The current design is batch-oriented (upload → process → download).
- A streaming architecture would allow real-time translation:
  - WebSocket endpoint for streaming audio input
  - Chunked ASR with streaming Whisper
  - Incremental translation as sentences complete
  - Streaming TTS output
- This requires significant architectural changes and is targeted for Phase 4+.

### Additional Languages

- The translation and TTS stages are designed with abstract interfaces to support additional
  language pairs.
- IndicTrans2 supports multiple Indic languages; adding new targets requires:
  - TTS model for the target language
  - Validation of translation quality for the new pair
  - Configuration for language-specific preprocessing

### Scalability

- The current single-process architecture is suitable for development and single-user deployment.
- Scaling options for production:
  - Multiple Uvicorn workers (process-based parallelism)
  - Task queue (Celery/RQ) for async pipeline execution
  - Model serving with dedicated inference servers (Triton, TorchServe)
  - Horizontal scaling behind a load balancer

---

## Technology Stack

| Component | Technology | Version | Verified on Python 3.14 | Verified on ARM64 |
|---|---|---|---|---|
| Language | Python | 3.14.x | N/A | Running under x64 emulation |
| Web Framework | FastAPI | ≥0.115 | ✅ Yes | ✅ Yes |
| ASGI Server | Uvicorn | ≥0.32 | ✅ Yes | ✅ Yes |
| Validation | Pydantic | ≥2.10 | ✅ Yes | ✅ Yes |
| Audio I/O | librosa | ≥0.10 | ⚠️ UNKNOWN | ⚠️ UNKNOWN |
| Audio I/O | soundfile | ≥0.12 | ⚠️ UNKNOWN | ⚠️ UNKNOWN |
| ML Runtime | PyTorch (CPU) | ≥2.0 | ⚠️ UNKNOWN | ⚠️ UNKNOWN |
| ASR | faster-whisper | ≥1.0 | ⚠️ UNKNOWN | ⚠️ UNKNOWN |
| Translation | IndicTrans2 | — | ⚠️ UNKNOWN | ⚠️ UNKNOWN |
| TTS | IndicF5 / Indic-Speak | — | ⚠️ UNKNOWN | ⚠️ UNKNOWN |
| VAD | Silero VAD | ≥5.1 | ⚠️ UNKNOWN | ⚠️ UNKNOWN |
| Speaker Embedding | resemblyzer / speechbrain | — | ⚠️ UNKNOWN | ⚠️ UNKNOWN |
| Config | pydantic-settings | ≥2.0 | ✅ Yes | ✅ Yes |
| Logging | structlog | ≥24.0 | ✅ Yes | ✅ Yes |
| Testing | pytest | ≥8.0 | ✅ Yes | ✅ Yes |
| System Dependency | FFmpeg | ≥6.0 | N/A (binary) | ⚠️ UNKNOWN |

> [!WARNING]
> Python 3.14 is very new (released 2025). Many ML libraries may not yet have pre-built wheels.
> Installation may require building from source, which can be slow and may fail on ARM64.
> All "UNKNOWN" entries above must be verified before committing to that dependency.
