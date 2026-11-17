"""Unit tests for ASR schemas and data contracts.

Owner: P2 — Speech Understanding Engineer
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from voxverse.genai.asr.schemas import (
    ASRModelInfo,
    LanguageValidationResult,
    TranscriptionResult,
    TranscriptionSegment,
)


def test_transcription_segment_valid():
    seg = TranscriptionSegment(
        id=0,
        seek=0,
        start=0.0,
        end=2.5,
        text="Hello world",
        avg_logprob=-0.25,
        no_speech_prob=0.01,
        compression_ratio=1.2,
    )
    assert seg.id == 0
    assert seg.start == 0.0
    assert seg.end == 2.5
    assert seg.text == "Hello world"


def test_language_validation_result():
    val = LanguageValidationResult(
        is_english=True,
        detected_language="en",
        language_probability=0.985,
        status="VERIFIED_ENGLISH",
        reason="High confidence English detected.",
    )
    assert val.is_english is True
    assert val.status == "VERIFIED_ENGLISH"


def test_language_validation_result_invalid_status():
    with pytest.raises(ValidationError):
        LanguageValidationResult(
            is_english=True,
            detected_language="en",
            status="INVALID_STATUS",  # type: ignore[arg-type]
        )


def test_asr_model_info():
    info = ASRModelInfo(
        model_name="tiny.en",
        backend="faster-whisper",
        compute_type="int8",
        is_multilingual=False,
        cpu_threads=4,
    )
    assert info.model_name == "tiny.en"
    assert info.compute_type == "int8"
    assert info.device == "cpu"


def test_transcription_result_schema():
    model_info = ASRModelInfo(
        model_name="tiny.en",
        backend="faster-whisper",
        compute_type="int8",
        is_multilingual=False,
        cpu_threads=4,
    )
    result = TranscriptionResult(
        text="Sample transcript text.",
        language="en",
        language_probability=1.0,
        duration_seconds=5.0,
        inference_duration_seconds=0.5,
        real_time_factor=0.1,
        model_info=model_info,
        segments=[
            TranscriptionSegment(
                id=0,
                start=0.0,
                end=5.0,
                text="Sample transcript text.",
            )
        ],
        peak_memory_mb=18.5,
    )
    assert result.text == "Sample transcript text."
    assert result.real_time_factor == 0.1
    assert len(result.segments) == 1
