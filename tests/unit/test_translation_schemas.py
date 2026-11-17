"""Unit tests for translation data models and schemas.

Owner: P2 — Speech Understanding Engineer
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from voxverse.genai.translation.schemas import (
    EvaluationMetricsResult,
    TranslationModelInfo,
    TranslationRequest,
    TranslationResult,
    TranslationSegment,
)


def test_translation_request_valid() -> None:
    req = TranslationRequest(
        text="Hello, world!",
        source_language="eng_Latn",
        target_language="tel_Telu",
        beam_size=4,
    )
    assert req.text == "Hello, world!"
    assert req.source_language == "eng_Latn"
    assert req.target_language == "tel_Telu"
    assert req.beam_size == 4


def test_translation_request_invalid_source_language() -> None:
    with pytest.raises(ValidationError) as exc:
        TranslationRequest(
            text="Hello",
            source_language="fra_Latn",
            target_language="tel_Telu",
        )
    assert "Unsupported source language" in str(exc.value)


def test_translation_request_invalid_target_language() -> None:
    with pytest.raises(ValidationError) as exc:
        TranslationRequest(
            text="Hello",
            source_language="eng_Latn",
            target_language="xyz_Abcd",
        )
    assert "Unsupported target language" in str(exc.value)


def test_translation_request_empty_text() -> None:
    with pytest.raises(ValidationError):
        TranslationRequest(
            text="",
            source_language="eng_Latn",
            target_language="tel_Telu",
        )


def test_translation_segment_model() -> None:
    seg = TranslationSegment(
        id=0,
        source_text="Hello",
        target_text="హలో",
        start=0.5,
        end=1.2,
    )
    assert seg.id == 0
    assert seg.source_text == "Hello"
    assert seg.target_text == "హలో"
    assert seg.start == 0.5
    assert seg.end == 1.2


def test_translation_result_model() -> None:
    info = TranslationModelInfo(
        model_name="prajdabre/rotary-indictrans2-en-indic-dist-200M",
        backend="transformers-rotary-indictrans2",
        device="cpu",
        compute_type="float32",
        parameter_count=211776512,
        max_sequence_length=256,
    )
    seg = TranslationSegment(id=0, source_text="Good morning", target_text="శుభోదయం")
    res = TranslationResult(
        source_text="Good morning",
        translated_text="శుభోదయం",
        source_language="eng_Latn",
        target_language="tel_Telu",
        duration_seconds=0.85,
        inference_duration_seconds=0.72,
        characters_per_second=14.1,
        model_info=info,
        segments=[seg],
        peak_memory_mb=1200.5,
    )
    assert res.translated_text == "శుభోదయం"
    assert len(res.segments) == 1
    assert res.peak_memory_mb == 1200.5


def test_evaluation_metrics_result() -> None:
    m = EvaluationMetricsResult(
        dataset_name="FLORES-200",
        split="devtest",
        sample_count=50,
        bleu=29.8,
        chrf=65.82,
        chrf_plus_plus=61.26,
        duration_seconds=821.69,
    )
    assert m.bleu == 29.8
    assert m.chrf_plus_plus == 61.26
    assert m.sample_count == 50
