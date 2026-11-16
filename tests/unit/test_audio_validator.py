"""Unit tests for the audio quality validator.

Covers:
- Valid speech-range audio
- Silence detection
- Clipping detection
- Short audio rejection
- Long audio rejection
- Invalid sample rate rejection
- Invalid channel count rejection
- Amplitude metrics calculation (RMS and Peak)
- Strict mode vs soft report

Owner: P1 — Audio and Speaker Profile Engineer
"""

from __future__ import annotations

import numpy as np
import pytest

from voxverse.genai.audio.exceptions import AudioValidationError
from voxverse.genai.audio.validator import AudioValidator


@pytest.fixture
def validator() -> AudioValidator:
    """Return an AudioValidator with standard test thresholds."""
    return AudioValidator(
        min_duration_seconds=0.5,
        max_duration_seconds=10.0,
        min_sample_rate=8000,
        max_sample_rate=48000,
        max_channels=2,
        silence_rms_threshold=0.001,
        clipping_peak_threshold=0.999,
    )


def test_validator_passes_valid_audio(validator: AudioValidator) -> None:
    """Test that a clean, properly-bounded 1-second sine wave passes validation."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    clean_audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    report = validator.validate(clean_audio, sr, strict=True)

    assert report.is_valid is True
    assert report.is_silent is False
    assert report.is_clipped is False
    assert report.duration_seconds == 1.0
    assert 0.34 < report.rms_amplitude < 0.36
    assert 0.49 < report.peak_amplitude < 0.51
    assert len(report.issues) == 0


def test_validator_detects_silence(validator: AudioValidator) -> None:
    """Test that digital silence (all zeros or near-zero) is flagged."""
    sr = 16000
    silent_audio = np.zeros(16000, dtype=np.float32)

    report = validator.validate(silent_audio, sr, strict=False)

    assert report.is_valid is False
    assert report.is_silent is True
    assert report.rms_amplitude == 0.0
    assert any("silent" in issue.lower() for issue in report.issues)


def test_validator_detects_clipping(validator: AudioValidator) -> None:
    """Test that samples pegged at maximum dynamic range are flagged as clipped."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    # Severely overdriven signal clipped at 1.0
    overdriven = 2.0 * np.sin(2 * np.pi * 440 * t)
    clipped_audio = np.clip(overdriven, -1.0, 1.0)

    report = validator.validate(clipped_audio, sr, strict=False)

    assert report.is_valid is False
    assert report.is_clipped is True
    assert report.peak_amplitude >= 0.999
    assert any("clipped" in issue.lower() for issue in report.issues)


def test_validator_flags_too_short_audio(validator: AudioValidator) -> None:
    """Test that audio shorter than min_duration is flagged."""
    sr = 16000
    # 0.2 seconds (min is 0.5s)
    short_audio = 0.5 * np.ones(int(0.2 * sr), dtype=np.float32)

    report = validator.validate(short_audio, sr, strict=False)

    assert report.is_valid is False
    assert any("shorter" in issue.lower() for issue in report.issues)


def test_validator_flags_too_long_audio(validator: AudioValidator) -> None:
    """Test that audio longer than max_duration is flagged."""
    sr = 16000
    # 12.0 seconds (max is 10.0s)
    long_audio = 0.5 * np.ones(int(12.0 * sr), dtype=np.float32)

    report = validator.validate(long_audio, sr, strict=False)

    assert report.is_valid is False
    assert any("exceeds" in issue.lower() for issue in report.issues)


def test_validator_flags_invalid_sample_rate(validator: AudioValidator) -> None:
    """Test that an out-of-bounds sample rate is rejected."""
    audio = np.zeros(1000, dtype=np.float32)

    # 4000 Hz is below min 8000 Hz
    report = validator.validate(audio, 4000, strict=False)

    assert report.is_valid is False
    assert any("sample rate" in issue.lower() for issue in report.issues)


def test_validator_flags_invalid_channels(validator: AudioValidator) -> None:
    """Test that audio with more channels than max_channels is flagged."""
    sr = 16000
    # 6-channel audio (max is 2)
    multichannel = np.zeros((16000, 6), dtype=np.float32)

    report = validator.validate(multichannel, sr, strict=False)

    assert report.is_valid is False
    assert any("channel count" in issue.lower() for issue in report.issues)


def test_validator_strict_mode_raises_exception(validator: AudioValidator) -> None:
    """Test that strict=True raises AudioValidationError on invalid audio."""
    sr = 16000
    silent_audio = np.zeros(16000, dtype=np.float32)

    with pytest.raises(AudioValidationError) as exc_info:
        validator.validate(silent_audio, sr, strict=True)

    assert exc_info.value.code == "AUDIO_QUALITY_FAILURE"
    assert len(exc_info.value.issues) > 0


def test_validator_empty_array_handling(validator: AudioValidator) -> None:
    """Test that an empty array returns a clean invalid report."""
    empty_audio = np.array([], dtype=np.float32)

    report = validator.validate(empty_audio, 16000, strict=False)

    assert report.is_valid is False
    assert report.duration_seconds == 0.0
    assert any("zero samples" in issue.lower() for issue in report.issues)
