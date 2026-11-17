"""Unit tests for ASRService using explicit mock engines.

Owner: P2 — Speech Understanding Engineer
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest
import soundfile as sf

from voxverse.genai.asr.base import BaseASREngine
from voxverse.genai.asr.exceptions import NonEnglishAudioError, TranscriptionError
from voxverse.genai.asr.schemas import (
    ASRModelInfo,
    LanguageValidationResult,
    TranscriptionResult,
    TranscriptionSegment,
)
from voxverse.genai.asr.service import ASRService


class MockASREngine(BaseASREngine):
    """Explicit mock ASR engine for unit tests."""

    def __init__(self, is_english: bool = True, detected_language: str = "en"):
        super().__init__(model_name="mock-model", cpu_threads=2)
        self.mock_is_english = is_english
        self.mock_detected_lang = detected_language
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def get_model_info(self) -> ASRModelInfo:
        return ASRModelInfo(
            model_name="mock-model",
            backend="faster-whisper",
            compute_type="int8",
            is_multilingual=True,
            cpu_threads=2,
        )

    def detect_language(self, audio_path) -> LanguageValidationResult:
        if self.mock_is_english:
            return LanguageValidationResult(
                is_english=True,
                detected_language="en",
                language_probability=0.99,
                status="VERIFIED_ENGLISH",
            )
        return LanguageValidationResult(
            is_english=False,
            detected_language=self.mock_detected_lang,
            language_probability=0.95,
            status="REJECTED_NON_ENGLISH",
            reason=f"Detected {self.mock_detected_lang}",
        )

    def transcribe(self, audio_path, **kwargs) -> TranscriptionResult:
        return TranscriptionResult(
            text="Mock transcribed speech.",
            language="en",
            language_probability=0.99,
            duration_seconds=3.0,
            inference_duration_seconds=0.25,
            real_time_factor=0.083,
            model_info=self.get_model_info(),
            segments=[
                TranscriptionSegment(
                    id=0,
                    start=0.0,
                    end=3.0,
                    text="Mock transcribed speech.",
                )
            ],
            peak_memory_mb=15.0,
        )


@pytest.fixture
def sample_wav(tmp_path: Path) -> Path:
    wav_path = tmp_path / "valid.wav"
    samples = np.zeros(16000 * 2, dtype=np.float32)
    sf.write(str(wav_path), samples, 16000, subtype="PCM_16")
    return wav_path


def test_service_successful_transcription(sample_wav: Path):
    engine = MockASREngine(is_english=True)
    service = ASRService(engine=engine)

    result = service.transcribe_file(sample_wav, validate_english=True)
    assert result.text == "Mock transcribed speech."
    assert result.real_time_factor == 0.083
    assert result.language_validation is not None
    assert result.language_validation.is_english is True


def test_service_rejects_non_english_audio(sample_wav: Path):
    engine = MockASREngine(is_english=False, detected_language="es")
    service = ASRService(engine=engine)

    with pytest.raises(NonEnglishAudioError) as exc_info:
        service.transcribe_file(sample_wav, validate_english=True)

    assert "Non-English audio rejected" in str(exc_info.value)
    assert exc_info.value.detected_language == "es"


def test_service_bypasses_validation_when_disabled(sample_wav: Path):
    engine = MockASREngine(is_english=False, detected_language="es")
    service = ASRService(engine=engine)

    # When validate_english=False, non-English audio is not rejected upfront
    result = service.transcribe_file(sample_wav, validate_english=False)
    assert result.text == "Mock transcribed speech."
