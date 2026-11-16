# VoxVerse

**AI-Powered English-to-Telugu Voice Translation**

> **Status:** Phase 0 — Foundation  
> **License:** TBD

---

## Overview

VoxVerse is a local, privacy-first voice translation platform that converts spoken English audio
into spoken Telugu audio. The entire pipeline — speech recognition, translation, and speech
synthesis — runs locally on your machine with no cloud API dependencies.

### Key Features (Planned)

- 🎙️ **English Speech Recognition** — powered by Whisper / Faster-Whisper
- 🔄 **English → Telugu Translation** — powered by AI4Bharat IndicTrans2
- 🔊 **Telugu Speech Synthesis** — powered by AI4Bharat IndicF5
- 🧑 **Voice Identity Preservation** — optional speaker embedding for voice similarity
- 🔒 **Privacy-First** — all processing is local; no data leaves your machine
- 🖥️ **CPU-Only** — runs on consumer hardware without a GPU

### Current Status

VoxVerse is in **Phase 0 (Foundation)**. The project structure, configuration, health-check API,
and documentation are in place. ML model integration is planned for subsequent phases.

---

## Team Structure

| Member | Role | Responsibilities |
|---|---|---|
| **P1** | Audio and Speaker Profile Engineer | Audio preprocessing, VAD, quality validation, voice enrollment, and speaker profile storage |
| **P2** | Speech Understanding Engineer | Whisper ASR, English language validation, IndicTrans2, and Telugu translation |
| **P3** | Voice Generation Engineer | IndicF5 research, alternative TTS research, speaker conditioning, Telugu generation, evaluation |
| **P4** | Pipeline Engineer | FastAPI backend, contracts, pipeline orchestration, job processing, metrics, and integration |

---

## Architecture Overview

VoxVerse processes audio through a sequential pipeline:

```
Audio Input → Validation → VAD → ASR → Translation → TTS → Audio Output
```

Each stage is an independent module with a defined interface, allowing models to be swapped or
upgraded without affecting the rest of the pipeline. See [ARCHITECTURE.md](./ARCHITECTURE.md)
for the full architecture document and [MODEL_DECISIONS.md](./MODEL_DECISIONS.md) for model
research and evaluation status.

---

## Prerequisites

| Requirement | Version | Status |
|---|---|---|
| **Python** | 3.14+ | ✅ Installed |
| **Git** | 2.x | ✅ Installed |
| **FFmpeg** | 6.x+ | ⚠️ **Not installed — required for audio processing** |

### Installing FFmpeg

FFmpeg is required for audio format conversion and loading. It must be installed separately and
available on your system PATH.

**Windows (winget):**
```bash
winget install Gyan.FFmpeg
```

**Windows (manual):**
1. Download from https://ffmpeg.org/download.html
2. Extract to a directory (e.g., `C:\ffmpeg`)
3. Add `C:\ffmpeg\bin` to your system PATH
4. Verify: `ffmpeg -version`

> [!WARNING]
> VoxVerse will not function correctly without FFmpeg. Audio loading and format conversion
> depend on it.

---

## Installation

### 1. Clone the Repository

```bash
git clone <repo-url>
cd VOXVERSE
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

> On macOS/Linux, use `source venv/bin/activate` instead.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> [!NOTE]
> Python 3.14 is very new. Some ML libraries may not have pre-built wheels and may require
> building from source. If installation fails for a specific package, check
> [MODEL_DECISIONS.md](./MODEL_DECISIONS.md) for known compatibility issues and alternatives.

---

## Running the Application

### 1. Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` to customize settings if needed. See `.env.example` for available options.

### 2. Start the Server

```bash
uvicorn voxverse.api.app:app --reload
```

The API server will start at `http://127.0.0.1:8000`.

### 3. Verify

```bash
curl http://127.0.0.1:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2026-11-16T10:30:00Z",
  "version": "0.1.0"
}
```

---

## Running Tests

```bash
pytest tests/ -v
```

Run tests with coverage:
```bash
pytest tests/ -v --cov=voxverse --cov-report=term-missing
```

---

## API Endpoints

| Method | Path | Description | Status |
|---|---|---|---|
| `GET` | `/health` | Service health check | ✅ Available |
| `GET` | `/api/v1/info` | System info and capabilities | ✅ Available |
| `POST` | `/api/v1/translate` | Full voice translation pipeline | 🔜 Planned |
| `POST` | `/api/v1/transcribe` | Speech-to-text only | 🔜 Planned |
| `POST` | `/api/v1/translate-text` | Text translation only | 🔜 Planned |
| `POST` | `/api/v1/synthesize` | Text-to-speech only | 🔜 Planned |

---

## Project Structure

```
VOXVERSE/
├── voxverse/                   # Main application package
│   ├── __init__.py
│   ├── api/                    # FastAPI application (P1)
│   │   ├── __init__.py
│   │   ├── app.py              # FastAPI app factory and routes
│   │   └── routes/             # Route definitions
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
│   └── utils/                  # Shared utilities
│       ├── __init__.py
│       ├── logging.py          # Structured logging setup
│       └── timing.py           # Performance timing
├── tests/                      # Test suite
│   ├── conftest.py
│   ├── test_api/
│   ├── test_audio/
│   ├── test_asr/
│   ├── test_translation/
│   ├── test_tts/
│   └── test_pipeline/
├── models/                     # Downloaded ML models (git-ignored)
├── data/                       # Runtime data (git-ignored)
├── scripts/                    # Utility scripts
├── .env.example                # Environment variable template
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── AGENTS.md                   # Engineering standards & agent operating charter
├── ARCHITECTURE.md             # System architecture
├── MODEL_DECISIONS.md          # Model research & decisions
└── README.md                   # This file
```

---

## Development Guidelines

All developers and automated agents must adhere to the engineering quality, authenticity, security, and review standards defined in [AGENTS.md](./AGENTS.md).

### Core Principles

- **Authenticity:** Never create fake model outputs, simulated voice cloning, or fabricated benchmarks.
- **Verification:** Every claimed feature must be verified with reproducible test evidence.
- **Modularity:** Respect P1–P4 ownership boundaries. Keep ML inference separate from API routes.
- **Privacy & Consent:** Explicit consent is required for speaker enrollment; profile deletion must be supported.

### Code Style

- **Type hints:** All function signatures must include type hints.
- **Docstrings:** All public functions and classes must have docstrings (Google style).
- **Formatting:** Use a consistent formatter (e.g., Black, Ruff).
- **Linting:** Use Ruff or Flake8 for linting.

### Git Workflow

- `main` — stable, tested code only
- `dev` — integration branch for feature merges
- Feature branches: `feature/<owner>/<description>` (e.g., `feature/p2/whisper-integration`)
- All merges require code review from at least one other team member.

### Testing

- Write unit tests for all new modules.
- Integration tests for pipeline stages should use small, synthetic test data.
- Do not commit large audio files or model weights to Git.
- Use `pytest` fixtures for shared test setup (see `tests/conftest.py`).

### Module Ownership

- Each team member owns specific modules (see [Team Structure](#team-structure)).
- Changes to shared modules (`config/`, `utils/`) should be coordinated.
- API changes (`api/`) must be discussed with the full team.

### Dependency Management

- Pin dependency versions in `requirements.txt`.
- Document any new dependency in the PR description.
- Verify new dependencies work on Python 3.14 before adding them.

---

## Disclaimer

> [!CAUTION]
> **Voice Data & Consent**
>
> VoxVerse processes voice recordings, which are biometric data and personally identifiable
> information (PII). If you use VoxVerse to process audio recordings of other people:
>
> - **You must obtain informed consent** from the speaker before recording and processing their
>   voice.
> - **You are responsible** for complying with all applicable privacy laws and regulations in
>   your jurisdiction (e.g., GDPR, CCPA, India's DPDPA).
> - **Voice cloning features** (speaker embedding / voice identity preservation) carry additional
>   ethical and legal obligations. Do not clone someone's voice without their explicit permission.
> - **VoxVerse does not collect, store, or transmit voice data** to any external service. All
>   processing is local. However, temporary files are created during processing — ensure your
>   local storage is appropriately secured.
>
> The VoxVerse team provides this software as-is for research and educational purposes. Users
> bear full responsibility for ethical and legal use.
