"""Explicit domain exceptions for the audio processing foundation.

Owner: P1 — Audio and Speaker Profile Engineer
"""

from __future__ import annotations


class AudioProcessingError(Exception):
    """Base exception for all audio engine processing failures."""

    def __init__(self, message: str, code: str = "AUDIO_PROCESSING_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class AudioDecodeError(AudioProcessingError):
    """Raised when an audio file cannot be decoded or has corrupted headers/streams."""

    def __init__(self, message: str, code: str = "AUDIO_DECODE_ERROR") -> None:
        super().__init__(message, code=code)


class AudioValidationError(AudioProcessingError):
    """Raised when audio attributes violate quality, duration, or channel constraints."""

    def __init__(
        self,
        message: str,
        code: str = "AUDIO_VALIDATION_ERROR",
        issues: list[str] | None = None,
    ) -> None:
        super().__init__(message, code=code)
        self.issues = issues or [message]


class FFmpegNotFoundError(AudioProcessingError):
    """Raised when container decoding requires FFmpeg but the binary is absent from PATH."""

    def __init__(
        self,
        message: str = "FFmpeg is required to process this audio format, but was not detected on system PATH.",
        code: str = "FFMPEG_NOT_FOUND",
    ) -> None:
        super().__init__(message, code=code)


class VADInferenceError(AudioProcessingError):
    """Raised when Voice Activity Detection inference fails."""

    def __init__(self, message: str, code: str = "VAD_INFERENCE_ERROR") -> None:
        super().__init__(message, code=code)
