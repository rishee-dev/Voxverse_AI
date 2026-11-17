"""Data models and schemas for Machine Translation (MT).

Owner: P2 — Speech Understanding Engineer (Translation & NLP)
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, field_validator


# Supported canonical language codes (FLORES-200 notation)
SUPPORTED_SOURCE_LANGUAGES = {"eng_Latn"}
SUPPORTED_TARGET_LANGUAGES = {
    "tel_Telu",  # Telugu
    "hin_Deva",  # Hindi
    "tam_Taml",  # Tamil
    "kan_Knda",  # Kannada
    "mal_Mlym",  # Malayalam
    "ben_Beng",  # Bengali
    "mar_Deva",  # Marathi
    "guj_Gujr",  # Gujarati
    "pan_Guru",  # Punjabi
    "ory_Orya",  # Odia
    "urd_Arab",  # Urdu
    "asm_Beng",  # Assamese
}


class TranslationSegment(BaseModel):
    """A single translated segment with optional timestamp alignments from ASR."""

    id: int = Field(..., description="Zero-based segment index.")
    source_text: str = Field(..., description="Source text segment in English.")
    target_text: str = Field(..., description="Translated text segment in Telugu script.")
    start: float | None = Field(
        default=None,
        description="Audio start offset in seconds (when mapped from ASR).",
    )
    end: float | None = Field(
        default=None,
        description="Audio end offset in seconds (when mapped from ASR).",
    )


class TranslationModelInfo(BaseModel):
    """Metadata describing the active translation engine and model parameters."""

    model_name: str = Field(..., description="Model identifier or Hugging Face repo.")
    backend: Literal["transformers-rotary-indictrans2", "ctranslate2", "mock"] = Field(
        ...,
        description="Underlying translation runtime engine.",
    )
    device: str = Field(default="cpu", description="Hardware execution device ('cpu').")
    compute_type: str = Field(
        default="float32",
        description="Precision or quantization type (e.g., 'float32', 'int8').",
    )
    parameter_count: int = Field(..., description="Total model parameter count.")
    max_sequence_length: int = Field(
        default=256,
        description="Maximum supported sequence length in tokens.",
    )


class TranslationRequest(BaseModel):
    """Request payload for text translation."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Source English text to translate.",
    )
    source_language: str = Field(
        default="eng_Latn",
        description="FLORES-200 source language code (default 'eng_Latn').",
    )
    target_language: str = Field(
        default="tel_Telu",
        description="FLORES-200 target language code (default 'tel_Telu').",
    )
    beam_size: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Beam search width for sequence generation.",
    )

    @field_validator("source_language")
    @classmethod
    def validate_source_language(cls, v: str) -> str:
        if v not in SUPPORTED_SOURCE_LANGUAGES:
            msg = f"Unsupported source language '{v}'. Supported: {sorted(SUPPORTED_SOURCE_LANGUAGES)}"
            raise ValueError(msg)
        return v

    @field_validator("target_language")
    @classmethod
    def validate_target_language(cls, v: str) -> str:
        if v not in SUPPORTED_TARGET_LANGUAGES:
            msg = f"Unsupported target language '{v}'. Supported: {sorted(SUPPORTED_TARGET_LANGUAGES)}"
            raise ValueError(msg)
        return v


class TranslationResult(BaseModel):
    """Complete translation response containing translated Telugu text and execution telemetry."""

    source_text: str = Field(..., description="Original input English text.")
    translated_text: str = Field(..., description="Translated Telugu text in Telugu Unicode script.")
    source_language: str = Field(default="eng_Latn", description="Source language code.")
    target_language: str = Field(default="tel_Telu", description="Target language code.")
    duration_seconds: float = Field(
        ...,
        description="Total end-to-end processing duration in seconds.",
    )
    inference_duration_seconds: float = Field(
        ...,
        description="Pure model forward pass duration in seconds.",
    )
    characters_per_second: float = Field(
        ...,
        description="Throughput in source characters processed per second.",
    )
    model_info: TranslationModelInfo = Field(
        ...,
        description="Technical specifications of the active engine.",
    )
    segments: list[TranslationSegment] = Field(
        default_factory=list,
        description="Segment-by-segment translations with alignment offsets if available.",
    )
    peak_memory_mb: float | None = Field(
        default=None,
        description="Peak process RAM during translation in Megabytes.",
    )


class EvaluationMetricsResult(BaseModel):
    """Evaluation metrics computed against reference translations."""

    dataset_name: str = Field(..., description="Evaluation benchmark dataset name.")
    split: str = Field(..., description="Dataset split (e.g. 'devtest').")
    sample_count: int = Field(..., description="Number of parallel sentence pairs evaluated.")
    bleu: float = Field(..., description="Corpus BLEU score (SacreBLEU standard).")
    chrf: float = Field(..., description="chrF score.")
    chrf_plus_plus: float = Field(..., description="chrF++ score (character n-grams + word n-grams).")
    duration_seconds: float = Field(..., description="Evaluation runtime in seconds.")
