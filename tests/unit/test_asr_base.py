"""Unit tests for BaseASREngine audio input validation.

Owner: P2 — Speech Understanding Engineer
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from voxverse.genai.asr.base import BaseASREngine
from voxverse.genai.asr.exceptions import AudioInputError
from voxverse.genai.asr.schemas import ASRModelInfo, LanguageValidationResult, TranscriptionResult


class DummyEngine(BaseASREngine):
    """Minimal concrete engine for testing base audio validation logic."""

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def is_loaded(self) -> bool:
        return True

    def get_model_info(self) -> ASRModelInfo:
        return ASRModelInfo(
            model_name="dummy",
            backend="faster-whisper",
            compute_type="int8",
            is_multilingual=False,
            cpu_threads=2,
        )

    def transcribe(self, audio_path, **kwargs) -> TranscriptionResult:
        raise NotImplementedError

    def detect_language(self, audio_path) -> LanguageValidationResult:
        raise NotImplementedError


def test_validate_and_load_audio_valid(tmp_path: Path):
    engine = DummyEngine(model_name="dummy")
    wav_path = tmp_path / "test.wav"
    samples = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, 16000, endpoint=False, dtype=np.float32))
    sf.write(str(wav_path), samples, 16000, subtype="PCM_16")

    path, audio_arr, dur, sr = engine.validate_and_load_audio(wav_path)
    assert path == wav_path.resolve()
    assert len(audio_arr) == 16000
    assert dur == pytest.approx(1.0)
    assert sr == 16000


def test_validate_and_load_audio_stereo_downmix(tmp_path: Path):
    engine = DummyEngine(model_name="dummy")
    wav_path = tmp_path / "stereo.wav"
    samples = np.zeros((16000, 2), dtype=np.float32)
    sf.write(str(wav_path), samples, 16000, subtype="PCM_16")

    path, audio_arr, dur, sr = engine.validate_and_load_audio(wav_path)
    assert audio_arr.ndim == 1
    assert len(audio_arr) == 16000


def test_validate_and_load_audio_missing_file():
    engine = DummyEngine(model_name="dummy")
    with pytest.raises(AudioInputError) as exc_info:
        engine.validate_and_load_audio("non_existent_file.wav")
    assert "Audio file not found" in str(exc_info.value)
    assert exc_info.value.code == "AUDIO_FILE_NOT_FOUND"


def test_validate_and_load_audio_empty_file(tmp_path: Path):
    engine = DummyEngine(model_name="dummy")
    empty_file = tmp_path / "empty.wav"
    empty_file.write_bytes(b"")

    with pytest.raises(AudioInputError) as exc_info:
        engine.validate_and_load_audio(empty_file)
    assert "empty (0 bytes)" in str(exc_info.value)
    assert exc_info.value.code == "EMPTY_AUDIO_FILE"


def test_validate_and_load_audio_corrupted_file(tmp_path: Path):
    engine = DummyEngine(model_name="dummy")
    corrupt_file = tmp_path / "corrupt.wav"
    corrupt_file.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt corrupted bytes header")

    with pytest.raises(AudioInputError) as exc_info:
        engine.validate_and_load_audio(corrupt_file)
    assert "Failed to decode" in str(exc_info.value)
    assert exc_info.value.code == "UNREADABLE_AUDIO_FILE"


def test_validate_and_load_audio_enforce_canonical_sr(tmp_path: Path):
    engine = DummyEngine(model_name="dummy")
    wav_path = tmp_path / "44k.wav"
    samples = np.zeros(44100, dtype=np.float32)
    sf.write(str(wav_path), samples, 44100, subtype="PCM_16")

    with pytest.raises(AudioInputError) as exc_info:
        engine.validate_and_load_audio(wav_path, enforce_canonical_sr=True)
    assert "does not match canonical 16000 Hz" in str(exc_info.value)
    assert exc_info.value.code == "INVALID_SAMPLE_RATE"
