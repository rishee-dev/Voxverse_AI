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


class MissingCheckpointError(ModelLoadError):
    """Raised when the specified TTS model checkpoint file is missing or inaccessible."""

    def __init__(self, checkpoint_path: str, model_id: str = "ai4bharat/IndicF5") -> None:
        message = f"TTS checkpoint for '{model_id}' was not found at '{checkpoint_path}'."
        super().__init__(message, details={"model_id": model_id, "checkpoint_path": checkpoint_path})
        self.checkpoint_path = checkpoint_path
        self.model_id = model_id


class MissingVocoderError(ModelLoadError):
    """Raised when the neural vocoder (e.g. Vocos) weights or configuration are missing."""

    def __init__(self, vocoder_id: str = "charactr/vocos-mel-24khz") -> None:
        message = f"Neural vocoder '{vocoder_id}' weights or configuration are missing."
        super().__init__(message, details={"vocoder_id": vocoder_id})
        self.vocoder_id = vocoder_id


class GatedRepositoryError(ModelLoadError):
    """Raised when the Hugging Face model repository is gated and authentication is missing."""

    def __init__(self, repo_id: str = "ai4bharat/IndicF5") -> None:
        message = (
            f"Repository '{repo_id}' is a gated model requiring terms acceptance and Hugging Face authentication. "
            "Please visit https://huggingface.co/" + repo_id + " to accept terms and configure HF_TOKEN."
        )
        super().__init__(message, details={"repo_id": repo_id})
        self.repo_id = repo_id


class MemoryAdmissionError(TTSError):
    """Raised when available system physical RAM is below the safety threshold for TTS inference."""

    def __init__(self, available_gb: float, required_gb: float) -> None:
        message = (
            f"Insufficient physical RAM for safe TTS execution: {available_gb:.2f} GB available, "
            f"minimum required threshold is {required_gb:.2f} GB. Execution stopped to protect system stability."
        )
        super().__init__(message, details={"available_gb": available_gb, "required_gb": required_gb})
        self.available_gb = available_gb
        self.required_gb = required_gb


class InferenceTimeoutError(TTSError):
    """Raised when speech synthesis inference exceeds the maximum permitted execution time."""

    def __init__(self, timeout_seconds: float) -> None:
        message = f"TTS synthesis timed out after {timeout_seconds:.1f} seconds."
        super().__init__(message, details={"timeout_seconds": timeout_seconds})
        self.timeout_seconds = timeout_seconds


class ConcurrentRequestError(TTSError):
    """Raised when multiple synthesis requests attempt concurrent execution on CPU."""

    def __init__(self, message: str = "Concurrent TTS requests are rejected under CPU safety limits.") -> None:
        super().__init__(message)


class InvalidAudioOutputError(TTSError):
    """Raised when generated audio fails validation (e.g. empty, NaN/Inf samples, clipping, corrupt header)."""

    def __init__(self, reason: str, path: str | None = None) -> None:
        message = f"Generated audio output validation failed: {reason}"
        super().__init__(message, details={"reason": reason, "path": path} if path else {"reason": reason})
        self.reason = reason
        self.path = path


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
