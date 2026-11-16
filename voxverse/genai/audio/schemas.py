"""Pydantic data models for audio processing, validation, and VAD.

Owner: P1 — Audio and Speaker Profile Engineer
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class AudioMetadata(BaseModel):
    """Complete technical metadata for an audio file."""

    original_format: str = Field(
        ...,
        description="Detected audio container/codec format (e.g., 'WAV', 'MP3', 'FLAC', 'OGG').",
        examples=["WAV"],
    )
    sample_rate: int = Field(
        ...,
        description="Sampling rate in Hertz of the source audio.",
        examples=[44100],
    )
    channels: int = Field(
        ...,
        description="Number of audio channels (1=mono, 2=stereo, etc.).",
        examples=[2],
    )
    bit_depth: int | None = Field(
        default=None,
        description="Bit depth per sample if applicable (e.g., 16, 24, 32).",
        examples=[16],
    )
    duration_seconds: float = Field(
        ...,
        description="Total duration of audio in seconds.",
        examples=[5.25],
    )
    frame_count: int = Field(
        ...,
        description="Total number of audio sample frames.",
        examples=[231525],
    )
    file_size_bytes: int = Field(
        ...,
        description="File size in bytes on disk.",
        examples=[926144],
    )
    normalized_sample_rate: int = Field(
        default=16000,
        description="Target sampling rate after normalization in Hertz.",
        examples=[16000],
    )
    normalized_channels: int = Field(
        default=1,
        description="Channel count after normalization (1=mono).",
        examples=[1],
    )
    normalized_format: str = Field(
        default="WAV",
        description="Standardized output container format.",
        examples=["WAV"],
    )


class AudioQualityReport(BaseModel):
    """Quality and integrity metrics measured from decoded audio."""

    is_valid: bool = Field(
        ...,
        description="True if audio meets all duration, amplitude, and formatting constraints.",
    )
    rms_amplitude: float = Field(
        ...,
        description="Root-Mean-Square (RMS) amplitude measured across samples (range [0.0, 1.0]).",
        examples=[0.125],
    )
    peak_amplitude: float = Field(
        ...,
        description="Maximum absolute peak sample amplitude (range [0.0, 1.0]).",
        examples=[0.85],
    )
    is_silent: bool = Field(
        ...,
        description="True if RMS amplitude falls below the silence threshold.",
    )
    is_clipped: bool = Field(
        ...,
        description="True if sample values exceed or reach saturation limits.",
    )
    duration_seconds: float = Field(
        ...,
        description="Duration of validated audio.",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="List of detected anomalies, warnings, or rejection reasons.",
    )


class SpeechSegment(BaseModel):
    """A single contiguous speech segment identified by VAD."""

    start_ms: float = Field(
        ...,
        description="Segment start timestamp in milliseconds.",
        examples=[250.0],
    )
    end_ms: float = Field(
        ...,
        description="Segment end timestamp in milliseconds.",
        examples=[1450.0],
    )
    confidence: float = Field(
        ...,
        description="Mean speech probability score for this segment (range [0.0, 1.0]).",
        examples=[0.92],
    )

    @property
    def duration_ms(self) -> float:
        """Duration of this speech segment in milliseconds."""
        return max(0.0, self.end_ms - self.start_ms)


class VADResult(BaseModel):
    """Structured output from Voice Activity Detection inference."""

    audio_duration_seconds: float = Field(
        ...,
        description="Total duration of processed audio in seconds.",
    )
    speech_duration_seconds: float = Field(
        ...,
        description="Cumulative duration of detected speech regions in seconds.",
    )
    speech_ratio: float = Field(
        ...,
        description="Ratio of speech to total duration (range [0.0, 1.0]).",
    )
    segments: list[SpeechSegment] = Field(
        default_factory=list,
        description="List of detected speech intervals.",
    )
    processing_time_ms: float = Field(
        ...,
        description="CPU execution time for VAD inference in milliseconds.",
    )
    real_time_factor: float = Field(
        ...,
        description="Ratio of processing time to audio duration (RTF < 1.0 indicates faster than real-time).",
    )
