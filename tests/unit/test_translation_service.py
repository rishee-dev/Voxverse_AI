"""Unit tests for TranslationService and ASR-to-MT handoff using mocked engine.

Owner: P2 — Speech Understanding Engineer
"""

from __future__ import annotations

import pytest

from voxverse.genai.asr.schemas import (
    ASRModelInfo,
    TranscriptionResult,
    TranscriptionSegment as ASRSegment,
)
from voxverse.genai.translation.base import BaseTranslationEngine
from voxverse.genai.translation.exceptions import EmptyInputError, InferenceError
from voxverse.genai.translation.metrics import compute_translation_metrics
from voxverse.genai.translation.schemas import TranslationModelInfo
from voxverse.genai.translation.service import TranslationService


class MockTranslationEngine(BaseTranslationEngine):
    """Predictable mock engine for isolated unit testing."""

    def __init__(self, should_fail: bool = False) -> None:
        super().__init__(model_name="mock-engine", device="cpu")
        self._loaded = False
        self.should_fail = should_fail
        self.call_count = 0

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def get_model_info(self) -> TranslationModelInfo:
        return TranslationModelInfo(
            model_name="mock-engine",
            backend="mock",
            device="cpu",
            compute_type="float32",
            parameter_count=1000,
            max_sequence_length=256,
        )

    def translate(
        self,
        text: str,
        *,
        source_language: str = "eng_Latn",
        target_language: str = "tel_Telu",
        beam_size: int = 4,
        max_length: int = 256,
    ) -> str:
        return self.translate_batch([text], beam_size=beam_size)[0]

    def translate_batch(
        self,
        texts: list[str],
        *,
        source_language: str = "eng_Latn",
        target_language: str = "tel_Telu",
        beam_size: int = 4,
        max_length: int = 256,
    ) -> list[str]:
        if self.should_fail:
            raise InferenceError("Mock hardware failure during inference")
        self.call_count += 1
        return [f"[TE: {t}]" for t in texts]


def test_translate_text_simple() -> None:
    engine = MockTranslationEngine()
    service = TranslationService(engine=engine)

    result = service.translate_text("Hello, how are you?")
    assert result.source_text == "Hello, how are you?"
    assert result.translated_text == "[TE: Hello, how are you?]"
    assert result.source_language == "eng_Latn"
    assert result.target_language == "tel_Telu"
    assert len(result.segments) == 1
    assert result.duration_seconds >= 0.0


def test_translate_text_empty_input_rejected() -> None:
    service = TranslationService(engine=MockTranslationEngine())
    with pytest.raises(EmptyInputError):
        service.translate_text("")
    with pytest.raises(EmptyInputError):
        service.translate_text("   \n\t  ")


def test_translate_text_multi_sentence() -> None:
    engine = MockTranslationEngine()
    service = TranslationService(engine=engine)

    text = "Hello there. Welcome to VoxVerse! Have a great day."
    result = service.translate_text(text)

    assert len(result.segments) == 3
    assert result.segments[0].source_text == "Hello there."
    assert result.segments[1].source_text == "Welcome to VoxVerse!"
    assert result.segments[2].source_text == "Have a great day."
    assert "[TE: Hello there.] [TE: Welcome to VoxVerse!] [TE: Have a great day.]" == result.translated_text


def test_translate_text_inference_error_propagates() -> None:
    failing_engine = MockTranslationEngine(should_fail=True)
    service = TranslationService(engine=failing_engine)

    with pytest.raises(InferenceError) as exc:
        service.translate_text("Test sentence")
    assert "Mock hardware failure" in str(exc.value)


def test_translate_transcription_handoff_from_asr() -> None:
    engine = MockTranslationEngine()
    service = TranslationService(engine=engine)

    asr_info = ASRModelInfo(
        model_name="base.en",
        backend="faster-whisper",
        compute_type="int8",
        is_multilingual=False,
        cpu_threads=4,
    )
    seg1 = ASRSegment(id=0, start=0.0, end=2.5, text="Welcome to the seminar.")
    seg2 = ASRSegment(id=1, start=2.5, end=5.8, text="Today we discuss voice translation.")
    asr_result = TranscriptionResult(
        text="Welcome to the seminar. Today we discuss voice translation.",
        language="en",
        duration_seconds=5.8,
        inference_duration_seconds=0.82,
        real_time_factor=0.141,
        model_info=asr_info,
        segments=[seg1, seg2],
    )

    result = service.translate_transcription(asr_result, target_language="tel_Telu")

    assert len(result.segments) == 2
    # Verify segment offsets and IDs are preserved from ASR
    assert result.segments[0].id == 0
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 2.5
    assert result.segments[0].source_text == "Welcome to the seminar."
    assert result.segments[0].target_text == "[TE: Welcome to the seminar.]"

    assert result.segments[1].id == 1
    assert result.segments[1].start == 2.5
    assert result.segments[1].end == 5.8
    assert result.segments[1].source_text == "Today we discuss voice translation."
    assert result.segments[1].target_text == "[TE: Today we discuss voice translation.]"

    assert (
        result.translated_text
        == "[TE: Welcome to the seminar.] [TE: Today we discuss voice translation.]"
    )


def test_translate_transcription_empty_rejected() -> None:
    service = TranslationService(engine=MockTranslationEngine())
    asr_info = ASRModelInfo(
        model_name="base.en",
        backend="faster-whisper",
        compute_type="int8",
        is_multilingual=False,
        cpu_threads=4,
    )
    empty_asr = TranscriptionResult(
        text="   ",
        language="en",
        duration_seconds=1.0,
        inference_duration_seconds=0.1,
        real_time_factor=0.1,
        model_info=asr_info,
        segments=[],
    )
    with pytest.raises(EmptyInputError):
        service.translate_transcription(empty_asr)


def test_compute_translation_metrics_calculation() -> None:
    hyps = ["హలో మీరు ఎలా ఉన్నారు", "శుభోదయం"]
    refs = ["హలో మీరు ఎలా ఉన్నారు?", "శుభోదయం!"]

    res = compute_translation_metrics(
        hypotheses=hyps,
        references=refs,
        dataset_name="Test-FLORES",
        split="devtest",
        duration_seconds=1.5,
    )
    assert res.dataset_name == "Test-FLORES"
    assert res.sample_count == 2
    assert res.bleu > 0.0
    assert res.chrf > 0.0
    assert res.chrf_plus_plus > 0.0


def test_compute_translation_metrics_length_mismatch() -> None:
    with pytest.raises(ValueError) as exc:
        compute_translation_metrics(["one"], ["one", "two"])
    assert "Length mismatch" in str(exc.value)
