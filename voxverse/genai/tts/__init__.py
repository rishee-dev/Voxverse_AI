"""VoxVerse GenAI TTS — Telugu speech synthesis and speaker conditioning (P3).

Owner: P3 — Voice Generation Engineer

Responsibilities:
- Telugu text-to-speech synthesis research and adapter development.
- Candidate evaluation (AI4Bharat IndicF5, Bodhan AI Indic-Speak, and CPU fallbacks).
- Speaker conditioning and cross-lingual voice identity preservation (English ref -> Telugu speech).
- Output audio post-processing (24 kHz / 16 kHz WAV, EBU R 128 loudness normalization).
- Objective speaker similarity measurement (ECAPA-TDNN) and MOS listening tests.

Anti-Simulation Invariant:
Voice identity preservation is a technical requirement to investigate and measure.
Never substitute ordinary TTS and claim that voice cloning was achieved.
"""

from __future__ import annotations

from voxverse.genai.tts.base import BaseTTSEngine
from voxverse.genai.tts.exceptions import (
    MissingConsentError,
    MissingReferenceTranscriptError,
    ModelLoadError,
    ReferenceAudioError,
    SynthesisError,
    TeluguScriptValidationError,
    TTSError,
    UnsupportedEngineError,
    VoiceProfileNotFoundError,
)
from voxverse.genai.tts.schemas import (
    SpeakerConditioningData,
    TTSModelInfo,
    TTSRequest,
    TTSResult,
)

__all__: list[str] = [
    "BaseTTSEngine",
    "MissingConsentError",
    "MissingReferenceTranscriptError",
    "ModelLoadError",
    "ReferenceAudioError",
    "SpeakerConditioningData",
    "SynthesisError",
    "TTSError",
    "TTSModelInfo",
    "TTSRequest",
    "TTSResult",
    "TeluguScriptValidationError",
    "UnsupportedEngineError",
    "VoiceProfileNotFoundError",
]
