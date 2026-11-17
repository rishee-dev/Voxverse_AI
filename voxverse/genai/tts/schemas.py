"""Data models and schemas for Text-to-Speech (TTS) and voice conditioning.

Owner: P3 — Voice Generation Engineer
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class SpeakerConditioningData(BaseModel):
    """Authorized speaker reference data required for zero-shot voice cloning."""

    voice_id: str = Field(..., description="Unique voice profile identifier.")
    owner_id: str = Field(..., description="Authenticated owner of the voice profile.")
    speaker_name: str = Field(..., description="Speaker legal or declared identity.")
    reference_audio_path: str = Field(
        ...,
        description="Path to authorized, validated canonical reference audio (WAV).",
    )
    reference_transcript: str = Field(
        ...,
        description="Verified transcript of the reference audio clip (mandatory for flow-matching).",
    )
    consent_hash: str = Field(..., description="Cryptographic SHA-256 hash of affirmative consent.")
    sample_rate: int = Field(default=16000, description="Reference audio sample rate in Hz.")


class TTSModelInfo(BaseModel):
    """Metadata describing the active speech synthesis engine and parameters."""

    model_name: str = Field(..., description="Model identifier (e.g., 'ai4bharat/IndicF5').")
    backend: Literal["indicf5-flow-matching", "vits-mms", "mock"] = Field(
        ...,
        description="Synthesis architecture and runtime backend.",
    )
    device: str = Field(default="cpu", description="Hardware execution device ('cpu').")
    output_sample_rate: int = Field(default=24000, description="Output audio sample rate in Hz.")
    parameter_count: int = Field(..., description="Total model parameter count.")
    supports_cross_lingual_cloning: bool = Field(
        default=True,
        description="Whether the engine supports zero-shot voice transfer across languages.",
    )


class TTSRequest(BaseModel):
    """Request payload for synthesizing speech from translated text."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Target text to synthesize in Telugu script.",
    )
    language: str = Field(default="tel_Telu", description="Target language code.")
    voice_id: str | None = Field(
        default=None,
        description="Optional voice profile ID for speaker-conditioned cloning. If None, default voice is used.",
    )
    nfe_steps: int = Field(
        default=32,
        ge=8,
        le=64,
        description="Number of flow-matching function evaluation steps (ODE solver steps).",
    )
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speaking rate multiplier.")


class TTSResult(BaseModel):
    """Complete speech synthesis response containing generated audio metadata and telemetry."""

    audio_path: str | None = Field(
        default=None,
        description="Filesystem path to the generated output WAV audio file.",
    )
    audio_duration_seconds: float = Field(
        ...,
        description="Duration of the generated speech audio in seconds.",
    )
    sample_rate: int = Field(default=24000, description="Audio sample rate of the output in Hz.")
    channels: int = Field(default=1, description="Audio channel count (1=mono).")
    inference_duration_seconds: float = Field(
        ...,
        description="Time taken to compute audio synthesis on CPU in seconds.",
    )
    real_time_factor: float = Field(
        ...,
        description="Real-Time Factor (inference_duration / audio_duration). Lower is faster.",
    )
    model_info: TTSModelInfo = Field(..., description="Active TTS model specifications.")
    speaker_similarity_score: float | None = Field(
        default=None,
        description="Objective speaker similarity score (cosine distance) if reference was provided.",
    )
    peak_memory_mb: float | None = Field(
        default=None,
        description="Peak process RAM in Megabytes during synthesis.",
    )
