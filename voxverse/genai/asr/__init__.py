"""VoxVerse ASR package — Automatic Speech Recognition for English audio (P2).

Owner: P2 — Speech Understanding Engineer
"""

from voxverse.genai.asr.base import BaseASREngine
from voxverse.genai.asr.exceptions import (
    ASRError,
    AudioInputError,
    IncompatibleComputeTypeError,
    ModelLoadError,
    NonEnglishAudioError,
    TranscriptionError,
)
from voxverse.genai.asr.faster_whisper_engine import FasterWhisperEngine
from voxverse.genai.asr.metrics import (
    calculate_cer,
    calculate_wer,
    normalize_transcript,
)
from voxverse.genai.asr.schemas import (
    ASRModelInfo,
    LanguageValidationResult,
    TranscriptionResult,
    TranscriptionSegment,
)
from voxverse.genai.asr.service import ASRService
from voxverse.genai.asr.whisper_engine import OpenAIWhisperEngine

__all__ = [
    "ASRError",
    "ASRModelInfo",
    "ASRService",
    "AudioInputError",
    "BaseASREngine",
    "FasterWhisperEngine",
    "IncompatibleComputeTypeError",
    "LanguageValidationResult",
    "ModelLoadError",
    "NonEnglishAudioError",
    "OpenAIWhisperEngine",
    "TranscriptionError",
    "TranscriptionResult",
    "TranscriptionSegment",
    "calculate_cer",
    "calculate_wer",
    "normalize_transcript",
]
