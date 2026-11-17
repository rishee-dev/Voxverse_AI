"""Data models and schemas for Automatic Speech Recognition (ASR).

Owner: P2 — Speech Understanding Engineer
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class TranscriptionSegment(BaseModel):
    """A timestamped segment of transcribed speech."""

    id: int = Field(..., description="Zero-based segment index.")
    seek: int = Field(default=0, description="Seek position in frames/samples.")
    start: float = Field(..., description="Segment start offset in seconds.")
    end: float = Field(..., description="Segment end offset in seconds.")
    text: str = Field(..., description="Transcribed text for this segment.")
    avg_logprob: float | None = Field(
        default=None,
        description="Average token log probability for this segment.",
    )
    no_speech_prob: float | None = Field(
        default=None,
        description="Probability that this segment contains no speech.",
    )
    compression_ratio: float | None = Field(
        default=None,
        description="Zlib compression ratio of the transcribed text.",
    )


class LanguageValidationResult(BaseModel):
    """Result of language detection and English validation."""

    is_english: bool = Field(
        ...,
        description="True if the speech is confirmed to be English.",
    )
    detected_language: str = Field(
        ...,
        description="ISO 639-1 language code detected or assumed (e.g., 'en', 'te').",
    )
    language_probability: float | None = Field(
        default=None,
        description="Confidence probability in the detected language (range [0.0, 1.0]).",
    )
    status: Literal[
        "VERIFIED_ENGLISH",
        "REJECTED_NON_ENGLISH",
        "UNCERTAIN",
        "ASSUMED_ENGLISH",
    ] = Field(
        ...,
        description="Validation outcome category.",
    )
    reason: str | None = Field(
        default=None,
        description="Explanation for rejection or assumption.",
    )


class ASRModelInfo(BaseModel):
    """Metadata describing the active ASR engine and runtime parameters."""

    model_name: str = Field(..., description="Model identifier (e.g., 'tiny.en', 'base.en').")
    backend: Literal["faster-whisper", "openai-whisper"] = Field(
        ...,
        description="ASR backend implementation engine.",
    )
    compute_type: str = Field(
        ...,
        description="Inference compute type or quantization (e.g., 'int8', 'float32').",
    )
    is_multilingual: bool = Field(
        ...,
        description="True if the model supports multi-language detection/transcription.",
    )
    device: str = Field(default="cpu", description="Hardware device ('cpu').")
    cpu_threads: int = Field(..., description="Number of CPU worker threads.")


class TranscriptionResult(BaseModel):
    """Complete result returned from speech transcription."""

    text: str = Field(..., description="Full concatenated transcription text.")
    language: str = Field(..., description="Detected or configured language code ('en').")
    language_probability: float | None = Field(
        default=None,
        description="Confidence in language detection if available.",
    )
    duration_seconds: float = Field(..., description="Audio duration in seconds.")
    inference_duration_seconds: float = Field(
        ...,
        description="Time taken to perform inference in seconds.",
    )
    real_time_factor: float = Field(
        ...,
        description="Real-Time Factor (inference_duration / duration_seconds). Lower is faster.",
    )
    model_info: ASRModelInfo = Field(..., description="Active engine and model specifications.")
    segments: list[TranscriptionSegment] = Field(
        default_factory=list,
        description="Timestamped breakdown of transcription segments.",
    )
    language_validation: LanguageValidationResult | None = Field(
        default=None,
        description="Language identification details if validation was performed.",
    )
    peak_memory_mb: float | None = Field(
        default=None,
        description="Peak RAM consumption during inference in Megabytes.",
    )
