"""Domain-specific exceptions for Text-to-Speech (TTS) synthesis and voice conditioning.

Owner: P3 — Voice Generation Engineer
"""

from __future__ import annotations


class TTSError(Exception):
    """Base exception for all text-to-speech synthesis errors."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ModelLoadError(TTSError):
    """Raised when the TTS model, flow-matching DiT, or vocoder fails to load into memory."""


class VoiceProfileNotFoundError(TTSError):
    """Raised when the specified voice profile does not exist in storage."""

    def __init__(self, voice_id: str) -> None:
        message = f"Voice profile with ID '{voice_id}' was not found in storage."
        super().__init__(message, details={"voice_id": voice_id})
        self.voice_id = voice_id


class MissingConsentError(TTSError):
    """Raised when affirmative speaker consent is missing, unverified, or revoked."""

    def __init__(self, voice_id: str, reason: str = "Active affirmative consent is required.") -> None:
        message = f"Voice profile '{voice_id}' cannot be used: {reason}"
        super().__init__(message, details={"voice_id": voice_id, "reason": reason})
        self.voice_id = voice_id
        self.reason = reason


class ReferenceAudioError(TTSError):
    """Raised when reference audio is missing, unreadable, corrupt, or fails duration/quality criteria."""

    def __init__(self, message: str, path: str | None = None) -> None:
        super().__init__(message, details={"path": path} if path else {})
        self.path = path


class MissingReferenceTranscriptError(TTSError):
    """Raised when reference audio transcript is missing or empty for models requiring it (e.g., IndicF5)."""

    def __init__(self, voice_id: str) -> None:
        message = (
            f"Voice profile '{voice_id}' does not have a verified reference transcript. "
            "Flow-matching TTS requires verbatim prompt text for cross-lingual synthesis."
        )
        super().__init__(message, details={"voice_id": voice_id})
        self.voice_id = voice_id


class SynthesisError(TTSError):
    """Raised when speech synthesis fails during neural network inference, ODE solving, or vocoding."""


class TeluguScriptValidationError(TTSError):
    """Raised when the input text fails Telugu script validation or contains unrenderable characters."""

    def __init__(self, text: str, reason: str) -> None:
        message = f"Telugu text failed synthesis validation: {reason}"
        super().__init__(message, details={"text_snippet": text[:50], "reason": reason})
        self.reason = reason


class UnsupportedEngineError(TTSError):
    """Raised when an unknown or unsupported TTS engine backend is requested."""

    def __init__(self, engine_name: str, supported: list[str]) -> None:
        message = f"TTS engine '{engine_name}' is not supported. Supported engines: {supported}"
        super().__init__(message, details={"engine_name": engine_name, "supported": supported})
        self.engine_name = engine_name
        self.supported = supported
