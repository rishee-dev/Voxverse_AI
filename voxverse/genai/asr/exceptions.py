"""Explicit domain exceptions for Automatic Speech Recognition (ASR).

Owner: P2 — Speech Understanding Engineer
"""

from __future__ import annotations


class ASRError(Exception):
    """Base exception for all ASR pipeline failures."""

    def __init__(self, message: str, code: str = "ASR_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class ModelLoadError(ASRError):
    """Raised when an ASR model cannot be loaded, downloaded, or initialized."""

    def __init__(self, message: str, code: str = "MODEL_LOAD_ERROR") -> None:
        super().__init__(message, code=code)


class AudioInputError(ASRError):
    """Raised when input audio is missing, empty, or violates format specifications."""

    def __init__(self, message: str, code: str = "AUDIO_INPUT_ERROR") -> None:
        super().__init__(message, code=code)


class TranscriptionError(ASRError):
    """Raised when transcription inference fails or produces unparseable output."""

    def __init__(self, message: str, code: str = "TRANSCRIPTION_ERROR") -> None:
        super().__init__(message, code=code)


class NonEnglishAudioError(ASRError):
    """Raised when audio is detected as non-English and strict English validation is enforced."""

    def __init__(
        self,
        message: str = "Audio was detected as non-English speech and was rejected.",
        detected_language: str | None = None,
        probability: float | None = None,
        code: str = "NON_ENGLISH_AUDIO_REJECTED",
    ) -> None:
        super().__init__(message, code=code)
        self.detected_language = detected_language
        self.probability = probability


class IncompatibleComputeTypeError(ASRError):
    """Raised when an unsupported compute type or quantization is requested for CPU."""

    def __init__(self, message: str, code: str = "INCOMPATIBLE_COMPUTE_TYPE") -> None:
        super().__init__(message, code=code)
