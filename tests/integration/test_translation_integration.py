"""Integration tests for real IndicTrans2 translation and ASR-to-MT handoff.

Owner: P2 — Speech Understanding Engineer
"""

from __future__ import annotations

import pytest

from voxverse.genai.asr.schemas import (
    ASRModelInfo,
    TranscriptionResult,
    TranscriptionSegment as ASRSegment,
)
from voxverse.genai.translation.indictrans2_engine import IndicTrans2Engine
from voxverse.genai.translation.service import TranslationService


@pytest.mark.integration
def test_real_indictrans2_inference() -> None:
    """Execute real English-to-Telugu translation inference on CPU."""
    engine = IndicTrans2Engine()
    service = TranslationService(engine=engine)

    source_text = "Good morning! How are you doing today?"
    result = service.translate_text(
        source_text,
        source_language="eng_Latn",
        target_language="tel_Telu",
        beam_size=4,
    )

    assert result.source_text == source_text
    assert result.source_language == "eng_Latn"
    assert result.target_language == "tel_Telu"
    # Verify the output contains genuine Telugu Unicode characters (\u0C00 - \u0C7F)
    assert any("\u0c00" <= c <= "\u0c7f" for c in result.translated_text)
    assert result.duration_seconds > 0.0
    assert result.inference_duration_seconds > 0.0
    assert result.characters_per_second > 0.0
    assert result.model_info.parameter_count > 200_000_000
    assert len(result.segments) >= 1


@pytest.mark.integration
def test_real_asr_to_translation_handoff() -> None:
    """Verify end-to-end handoff from ASR transcription segments to Telugu translation."""
    engine = IndicTrans2Engine()
    service = TranslationService(engine=engine)

    asr_info = ASRModelInfo(
        model_name="base.en",
        backend="faster-whisper",
        compute_type="int8",
        is_multilingual=False,
        cpu_threads=4,
    )
    seg1 = ASRSegment(id=0, start=0.0, end=2.4, text="The scientific conference starts on Monday.")
    seg2 = ASRSegment(id=1, start=2.4, end=4.8, text="We look forward to meeting you.")
    asr_result = TranscriptionResult(
        text="The scientific conference starts on Monday. We look forward to meeting you.",
        language="en",
        duration_seconds=4.8,
        inference_duration_seconds=0.68,
        real_time_factor=0.142,
        model_info=asr_info,
        segments=[seg1, seg2],
    )

    translation_result = service.translate_transcription(
        asr_result,
        target_language="tel_Telu",
        beam_size=4,
    )

    assert len(translation_result.segments) == 2
    # Verify segment offsets are preserved from ASR
    assert translation_result.segments[0].id == 0
    assert translation_result.segments[0].start == 0.0
    assert translation_result.segments[0].end == 2.4
    assert any("\u0c00" <= c <= "\u0c7f" for c in translation_result.segments[0].target_text)

    assert translation_result.segments[1].id == 1
    assert translation_result.segments[1].start == 2.4
    assert translation_result.segments[1].end == 4.8
    assert any("\u0c00" <= c <= "\u0c7f" for c in translation_result.segments[1].target_text)

    # Full translated text is concatenated Telugu script
    assert any("\u0c00" <= c <= "\u0c7f" for c in translation_result.translated_text)
