"""Unit tests for the Silero VAD engine on ONNX Runtime CPU.

Covers:
- Model loading and session initialization
- Rejection of non-16 kHz audio (protocol contract)
- Processing digital silence (0 segments, 0 speech ratio)
- Empty audio handling
- Real-time factor calculation and processing metrics
- Missing model file error handling

Owner: P1 — Audio and Speaker Profile Engineer
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voxverse.genai.audio.exceptions import VADInferenceError
from voxverse.genai.audio.vad import SileroVADDetector


@pytest.fixture
def vad_detector() -> SileroVADDetector:
    """Return a SileroVADDetector loaded from the verified local ONNX model."""
    model_path = Path("models/silero_vad.onnx")
    if not model_path.exists():
        pytest.skip("models/silero_vad.onnx not present; skipping VAD tests")
    return SileroVADDetector(model_path=model_path)


def test_vad_initialization_missing_model(tmp_path: Path) -> None:
    """Test that pointing to a non-existent model file raises VADInferenceError."""
    bad_path = tmp_path / "non_existent.onnx"

    with pytest.raises(VADInferenceError) as exc_info:
        SileroVADDetector(model_path=bad_path)

    assert exc_info.value.code == "MODEL_NOT_FOUND"


def test_vad_rejects_non_16k_sample_rate(vad_detector: SileroVADDetector) -> None:
    """Test that Silero VAD strictly enforces 16,000 Hz input sample rate."""
    audio = np.zeros(8000, dtype=np.float32)

    with pytest.raises(VADInferenceError) as exc_info:
        vad_detector.detect_speech(audio, sample_rate=8000)

    assert exc_info.value.code == "INVALID_SAMPLE_RATE"


def test_vad_handles_empty_audio(vad_detector: SileroVADDetector) -> None:
    """Test that an empty array returns a clean, zeroed VADResult."""
    empty_audio = np.array([], dtype=np.float32)

    result = vad_detector.detect_speech(empty_audio, sample_rate=16000)

    assert result.audio_duration_seconds == 0.0
    assert result.speech_duration_seconds == 0.0
    assert result.speech_ratio == 0.0
    assert len(result.segments) == 0


def test_vad_processes_silence_correctly(vad_detector: SileroVADDetector) -> None:
    """Test that digital silence produces zero speech segments and zero false positives."""
    # 2.0 seconds of digital silence
    sr = 16000
    silence = np.zeros(int(2.0 * sr), dtype=np.float32)

    result = vad_detector.detect_speech(silence, sample_rate=sr)

    assert result.audio_duration_seconds == 2.0
    assert result.speech_duration_seconds == 0.0
    assert result.speech_ratio == 0.0
    assert len(result.segments) == 0
    assert result.processing_time_ms > 0.0
    assert result.real_time_factor < 1.0, "VAD inference should run much faster than real-time!"


def test_vad_performance_and_rtf_metrics(vad_detector: SileroVADDetector) -> None:
    """Test that processing duration, RTF, and metrics are accurately tracked."""
    # 1.0 second of audio
    sr = 16000
    audio = np.zeros(sr, dtype=np.float32)

    result = vad_detector.detect_speech(audio, sample_rate=sr)

    assert result.audio_duration_seconds == 1.0
    assert result.processing_time_ms > 0.0
    assert 0.0 <= result.speech_ratio <= 1.0
    # Expected RTF on this machine is < 0.05
    assert result.real_time_factor < 0.1
