# VOXVERSE — ENGINEERING STANDARDS & AGENT OPERATING CHARTER

> **Status:** ACTIVE & MANDATORY  
> **Scope:** Applies to every automated coding agent, human contributor, code review, API contract, and UI component across VoxVerse.  
> **Version:** 1.0.0 (Established Phase 0)

---

## 1. Product Identity & Authenticity

VoxVerse is an engineering-first, production-oriented multilingual AI voice translation platform. It translates English speech to Telugu while actively researching and evaluating voice identity preservation under CPU constraints.

### 1.1 Genuine Product Standards
- VoxVerse must be engineered as a purpose-built software product.
- It must **never** resemble a generic AI starter kit, tutorial template, or decorative demo assembled without technical rationale.
- AI-assisted development is permitted, but the resulting code must be clean, maintainable, idiomatically structured, fully tested, and thoroughly explainable by any engineer on the team.
- **Transparency:** Never misrepresent automated or AI-assisted generation as manual human craftsmanship. Maintain honest engineering documentation.

### 1.2 Anti-Simulation Invariants
- **Never produce fake model outputs:** Do not mock model responses and label them as real inference.
- **Never simulate voice cloning:** Voice identity preservation is an open technical challenge on CPU. Never route audio through ordinary text-to-speech (TTS) and claim that voice cloning was achieved.
- **Never fabricate benchmark results:** Every reported latency, memory footprint, Word Error Rate (WER), BLEU score, or speaker similarity metric must be reproduced on actual hardware with logged evidence.
- **Never mask stubs as implementations:** If a model adapter, audio preprocessor, or translation engine is unintegrated or experimental, label it explicitly with `FeatureStatus.NOT_IMPLEMENTED` or `FeatureStatus.EXPERIMENTAL`.
- **Reproducibility:** Every claimed milestone or feature must come with clear commands to reproduce the outcome locally.

---

## 2. Engineering & Code Quality Standards

Every line of code committed to VoxVerse must adhere to production-grade Python conventions.

### 2.1 Python Standards
- **Idiomatic Python 3.11+:** Use modern Python idioms (pattern matching, context managers, dataclasses/Pydantic, path-handling via `pathlib.Path`).
- **Strict Typing:** All function signatures, method parameters, and return types must include explicit Python type hints. Strive for zero `Any` unless interfacing with dynamic third-party libraries.
- **Pydantic for Data Validation:** All external inputs, configuration settings, and API request/response bodies must be validated using Pydantic v2 schemas.
- **Single Responsibility:** Keep functions and classes focused on one task. Break functions longer than 50 lines into well-named private helpers.
- **Thoughtful Abstractions:** Use abstractions (e.g., abstract base classes or protocols) only when there is a concrete need to swap engines (such as switching between `WhisperEngine` and `FasterWhisperEngine`). Avoid redundant layers of indirection that obscure data flow.
- **Explicit Imports:** Avoid wildcard imports (`from module import *`). Avoid unused imports. Group imports logically: standard library, third-party packages, and first-party `voxverse` modules.
- **No Dead Code:** Never leave commented-out blocks, scratch scripts, or unexplained temporary debug prints in production modules.

### 2.2 Error Handling & Logging
- **Explicit Exceptions:** Catch specific exceptions (`ValueError`, `FileNotFoundError`, `HTTPException`). Never use bare `except:` or catch-all `except Exception:` unless in top-level middleware where the error is logged with full context.
- **Structured Logging:** Use the centralized logger from `voxverse.config.logging`. Never use `print()` statements for logging application state or errors.
- **Correlation IDs:** Every log record and error payload must propagate the per-request `request_id` context to enable end-to-end tracing across pipeline stages.
- **No Leaked Stack Traces:** Public API error responses must adhere to `ErrorResponse` (RFC 7807 format) and must never expose internal stack traces, system paths, or environment secrets to clients.

---

## 3. Product Design & UI Principles

The user-facing interface of VoxVerse must feel like a precision audio workstation, not a marketing website.

### 3.1 Interface Tenets
- **Strong Typography:** Clear hierarchy using clean, legible sans-serif and monospace typefaces suitable for technical data and Indian scripts (Unicode Telugu rendering).
- **Restrained Color Palette:** Neutral base colors (deep slate, muted grays, crisp borders) with purposeful, high-contrast accent colors reserved for state changes (recording, processing, playback, errors).
- **Purposeful Audio Controls:** Native, responsive audio visualizers, waveform scrubbers, playback speed toggles, and clear duration/clipping indicators.
- **Meaningful States:** Every component must provide deliberate, functional empty states, progressive loading states with real stage progress, clear success messages, and actionable error states.
- **Accessibility:** High color contrast (WCAG AA compliant), full keyboard navigability, and clear ARIA labeling for all interactive controls.

### 3.2 What to Avoid
- **No generic AI dashboard templates:** Avoid cookie-cutter admin templates with meaningless stat tiles.
- **No decorative excess:** Avoid gratuitous background gradients, heavy glassmorphism, floating blur orbs, and pulsating neon glows.
- **No decorative charts:** Never display random graphs or activity meters that do not represent real system metrics.
- **No fake real-time activity:** Do not simulate artificial progress bars or mock terminal tickers. Progress must map 1:1 to actual pipeline stage execution.
- **No emoji-heavy professional interfaces:** Use clean, standard technical iconography instead of informal emojis in primary UI controls.

---

## 4. Documentation Standards

Documentation in VoxVerse must serve as an accurate engineering record of technical decisions, trade-offs, and empirical findings.

### 4.1 Decision Record Structure
For every major architectural or model decision, document:
1. **The Problem:** The specific technical requirement, hardware constraint, or performance bottleneck.
2. **Selected Approach:** The concrete algorithm, model variant, or architectural pattern chosen.
3. **Alternatives Considered:** Other options evaluated (e.g., OpenAI Whisper vs. Faster-Whisper INT8 CPU).
4. **Rationale:** Why the selected approach was chosen based on evidence and hardware feasibility.
5. **Known Limitations:** Current bottlenecks, CPU execution latency, memory limits, or missing capabilities.
6. **Reproduction Steps:** Exact commands, sample audio, and environment parameters required to verify the decision.

### 4.2 Tone & Integrity
- Maintain an objective, disciplined engineering tone.
- Do not write promotional claims or marketing assertions.
- Clearly differentiate between **verified facts** (tested on local hardware) and **candidate hypotheses** (documented in papers or community benchmarks).

---

## 5. Security, Privacy & Voice Rights

Voice audio and voice prints are biometric personal information. VoxVerse enforces strict safeguards across every pipeline stage.

### 5.1 Voice Enrollment & Consent
- **Mandatory Consent:** Voice enrollment and speaker reference storage must require explicit, active consent from the speaker (`VOXVERSE_SPEAKER_CONSENT_REQUIRED=true`).
- **Audit Metadata:** Every enrolled speaker profile must store a cryptographic hash, enrollment timestamp, consent declaration, and reference audio quality score.
- **Deletion Lifecycle:** The system must provide complete, irreversible deletion of speaker profiles, reference audio, and derived embeddings upon request (`DELETE /api/v1/voices/{id}`).

### 5.2 Confidentiality & Version Control
- **Never commit audio recordings:** Raw or preprocessed voice recordings must never be committed to Git.
- **Never commit speaker profiles or embeddings:** All `.npy`, `.npz`, and serialized embedding files are strictly ignored via `.gitignore`.
- **Never commit model weights or checkpoints:** Model binaries (`.pt`, `.bin`, `.safetensors`, `.onnx`, `.gguf`) belong in local cache directories, never in version control.
- **Never commit secrets:** API keys, certificates, private tokens, and local `.env` files must remain strictly local. Only `.env.example` with safe defaults is tracked.
- **Never commit local databases:** Development SQLite files (`*.db`, `*.sqlite`) must remain untracked.

---

## 6. Code Review & Verification Standards

Before marking any task, pull request, or agent turn as complete, the author must execute and document the following 8-step review checklist:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   PRE-COMPLETION CODE REVIEW CHECKLIST                  │
├────────────────────────────────────────────────────────────────────────┤
│ 1. INSPECT FILES       Verify every added, edited, or deleted file.    │
│ 2. RUN TESTS           Execute full test suite (pytest tests/ -v).     │
│ 3. CHECK DEPENDENCIES  Ensure dependencies are pinned and documented.  │
│ 4. VERIFY ERRORS       Confirm explicit exception handling & logging.  │
│ 5. AUDIT SECURITY      Verify no secrets, audio, or embeddings leaked. │
│ 6. EVALUATE EFFICIENCY Confirm CPU suitability & memory boundaries.    │
│ 7. REPORT RESULTS      Include actual terminal output from tests.      │
│ 8. NOTE LIMITATIONS    Document any remaining technical unknowns.      │
└────────────────────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **Rule of Verification:** Never state or imply that a feature works until automated tests or live runtime execution have succeeded and their output has been verified.

---

## 7. Team Collaboration & Ownership Boundaries

VoxVerse maintains four distinct engineering ownership domains. Each engineer (or automated agent acting in that role) owns their respective modules and must respect neighboring interfaces:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            TEAM ROLES & DOMAINS                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  P1 — Audio and Speaker Profile Engineer                                     │
│  • Domain: voxverse/genai/audio/, voxverse/genai/speaker/                    │
│  • Owns: Audio format validation, resampling, Silero VAD, quality estimation │
│    voice enrollment, consent recording, and profile deletion lifecycle.      │
│                                                                              │
│  P2 — Speech Understanding Engineer                                          │
│  • Domain: voxverse/genai/asr/, voxverse/genai/translation/                  │
│  • Owns: Whisper / faster-whisper ASR, English validation, IndicTrans2 en→te │
│    translation, Telugu text normalization, and tokenization.                 │
│                                                                              │
│  P3 — Voice Generation Engineer                                              │
│  • Domain: voxverse/genai/tts/, voxverse/genai/evaluation/                   │
│  • Owns: IndicF5 / Indic-Speak TTS research, CPU fallback adapters, speaker   │
│    conditioning, voice preservation measurement, and MOS evaluation.         │
│                                                                              │
│  P4 — Pipeline Engineer                                                      │
│  • Domain: voxverse/api/, voxverse/config/, voxverse/storage/,               │
│    voxverse/genai/pipeline/                                                  │
│  • Owns: FastAPI routes, OpenAPI schemas, pipeline orchestration, batch job  │
│    processing, telemetry metrics, and filesystem/SQLite storage layer.       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 7.1 Cross-Module Collaboration Rules
- **Contract Stability:** Do not modify another engineer's input/output schemas or function signatures without explicit justification and test updates.
- **Loose Coupling:** ML inference engines must remain decoupled from the HTTP API routes. Routes interact with engines exclusively through typed interfaces and dependency injection.
- **Shared Modules:** Updates to `voxverse/config/`, shared schemas in `voxverse/api/schemas.py`, and top-level configuration require review across all affected roles.
