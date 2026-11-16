"""VoxVerse GenAI Audio — Audio preprocessing, validation, normalization, and VAD (P1).

Owner: P1 — Audio and Speaker Profile Engineer
"""

from __future__ import annotations

from voxverse.genai.audio.decoder import AudioDecoder, is_ffmpeg_available
from voxverse.genai.audio.exceptions import (
    AudioDecodeError,
    AudioProcessingError,
    AudioValidationError,
    FFmpegNotFoundError,
    VADInferenceError,
)
from voxverse.genai.audio.normalizer import AudioNormalizer
from voxverse.genai.audio.schemas import (
    AudioMetadata,
    AudioQualityReport,
    SpeechSegment,
    VADResult,
)
from voxverse.genai.audio.validator import AudioValidator
from voxverse.genai.audio.vad import SileroVADDetector

__all__: list[str] = [
    "AudioDecoder",
    "AudioNormalizer",
    "AudioValidator",
    "SileroVADDetector",
    "AudioMetadata",
    "AudioQualityReport",
    "SpeechSegment",
    "VADResult",
    "AudioProcessingError",
    "AudioDecodeError",
    "AudioValidationError",
    "FFmpegNotFoundError",
    "VADInferenceError",
    "is_ffmpeg_available",
]
