"""Integration tests running real ASR model inference on actual speech audio.

Owner: P2 — Speech Understanding Engineer
"""

from __future__ import annotations

from pathlib import Path
import pytest

from voxverse.genai.asr.faster_whisper_engine import FasterWhisperEngine
from voxverse.genai.asr.service import ASRService
from voxverse.genai.asr.whisper_engine import OpenAIWhisperEngine
from voxverse.genai.audio.normalizer import AudioNormalizer

SAMPLE_WAV = Path("data/benchmark/librispeech/wav_16k/6930-75918-0000.wav")
SAMPLE_FLAC = Path("data/benchmark/librispeech/raw/6930-75918-0000.flac")


@pytest.mark.skipif(not SAMPLE_WAV.exists(), reason="LibriSpeech test sample not found")
def test_faster_whisper_real_inference():
    """Verify real faster-whisper INT8 CPU inference on LibriSpeech speech."""
    engine = FasterWhisperEngine(model_name="tiny.en", compute_type="int8")
    assert not engine.is_loaded()

    engine.load()
    assert engine.is_loaded()

    result = engine.transcribe(SAMPLE_WAV)

    assert result.duration_seconds == pytest.approx(3.5, abs=0.1)
    assert result.real_time_factor < 1.0  # Must be faster than real-time
    assert len(result.text) > 0
    assert "concord" in result.text.lower()
    assert len(result.segments) > 0

    for seg in result.segments:
        assert seg.start >= 0.0
        assert seg.end >= seg.start
        assert len(seg.text) > 0

    engine.unload()
    assert not engine.is_loaded()


@pytest.mark.skipif(not SAMPLE_WAV.exists(), reason="LibriSpeech test sample not found")
def test_openai_whisper_real_inference():
    """Verify real OpenAI Whisper CPU inference on LibriSpeech speech."""
    engine = OpenAIWhisperEngine(model_name="tiny.en")
    assert not engine.is_loaded()

    engine.load()
    assert engine.is_loaded()

    result = engine.transcribe(SAMPLE_WAV)

    assert result.duration_seconds == pytest.approx(3.5, abs=0.1)
    assert result.real_time_factor < 1.0
    assert len(result.text) > 0
    assert "concord" in result.text.lower()

    engine.unload()
    assert not engine.is_loaded()


@pytest.mark.skipif(not SAMPLE_FLAC.exists(), reason="LibriSpeech raw FLAC sample not found")
def test_p1_to_p2_end_to_end_pipeline(tmp_path: Path):
    """Verify end-to-end integration: P1 AudioNormalizer feeding P2 ASRService."""
    # 1. P1 Normalize raw FLAC to standardized 16 kHz mono 16-bit PCM WAV
    normalizer = AudioNormalizer()
    normalized_wav = tmp_path / "p1_output.wav"
    dest, meta, quality = normalizer.normalize(SAMPLE_FLAC, normalized_wav)

    assert dest.exists()
    assert meta.normalized_sample_rate == 16000
    assert meta.normalized_channels == 1
    assert quality.is_valid is True

    # 2. P2 Transcribe using ASRService
    engine = FasterWhisperEngine(model_name="tiny.en", compute_type="int8")
    service = ASRService(engine=engine)

    result = service.transcribe_file(dest, validate_english=True)
    assert len(result.text) > 0
    assert "concord" in result.text.lower()
    assert result.language == "en"
    assert result.real_time_factor < 0.5  # Sub-realtime

    engine.unload()
