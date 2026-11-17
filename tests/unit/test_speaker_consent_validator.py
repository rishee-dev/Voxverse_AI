"""Unit tests for speaker affirmative consent verification and audio enrollment validation.

Owner: P1 — Audio and Speaker Profile Engineer
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from voxverse.genai.speaker.exceptions import (
    ConsentRequiredError,
    EnrollmentValidationError,
    InvalidConsentError,
)
from voxverse.genai.speaker.schemas import (
    ConsentPurpose,
    VoiceEnrollmentRequest,
)
from voxverse.genai.speaker.service import SpeakerProfileService
from voxverse.genai.speaker.validator import EnrollmentValidator
from voxverse.storage.voice_storage import VoiceProfileStorage


@pytest.fixture
def synthetic_speech_wav(tmp_path: Path) -> Path:
    """Provide verified LibriSpeech 3.5s speech audio."""
    librispeech_sample = Path("data/benchmark/librispeech/wav_16k/6930-75918-0000.wav")
    wav_path = tmp_path / "valid_speech.wav"
    if librispeech_sample.exists():
        wav_path.write_bytes(librispeech_sample.read_bytes())
    else:
        # Fallback 4.0s
        sr = 16000
        duration = 4.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
        audio = (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
        sf.write(str(wav_path), audio, sr, subtype="PCM_16")
    return wav_path


@pytest.fixture
def synthetic_silence_wav(tmp_path: Path) -> Path:
    """Generate 4.0s of digital silence."""
    wav_path = tmp_path / "silence.wav"
    sf.write(str(wav_path), np.zeros(16000 * 4, dtype=np.float32), 16000, subtype="PCM_16")
    return wav_path


@pytest.fixture
def synthetic_clipped_wav(tmp_path: Path) -> Path:
    """Generate clipped/saturated audio."""
    wav_path = tmp_path / "clipped.wav"
    clipped = np.ones(16000 * 4, dtype=np.float32) * 1.0
    sf.write(str(wav_path), clipped, 16000, subtype="PCM_16")
    return wav_path


@pytest.fixture
def synthetic_short_wav(tmp_path: Path) -> Path:
    """Generate audio shorter than 3.0s minimum threshold."""
    wav_path = tmp_path / "short.wav"
    sf.write(str(wav_path), np.sin(2 * np.pi * 440 * np.linspace(0, 1.5, 16000 * 1, endpoint=False, dtype=np.float32)), 16000, subtype="PCM_16")
    return wav_path


# ── Consent Invariant Tests ──────────────────────────────────────────────────


def test_consent_missing_affirmation(tmp_path: Path, synthetic_speech_wav: Path):
    storage = VoiceProfileStorage(base_dir=tmp_path / "voices")
    service = SpeakerProfileService(storage=storage)

    req = VoiceEnrollmentRequest(
        owner_id="user_123",
        display_name="Primary Voice",
        consent_confirmed=False,  # Violated
        ownership_confirmed=True,
        purposes=[ConsentPurpose.STORE_VOICE_REFERENCE],
    )

    with pytest.raises(ConsentRequiredError) as exc_info:
        service.enroll_voice(synthetic_speech_wav, req)
    assert "explicit affirmative consent" in str(exc_info.value)
    assert exc_info.value.code == "CONSENT_CONFIRMATION_REQUIRED"


def test_consent_missing_ownership_affirmation(tmp_path: Path, synthetic_speech_wav: Path):
    storage = VoiceProfileStorage(base_dir=tmp_path / "voices")
    service = SpeakerProfileService(storage=storage)

    req = VoiceEnrollmentRequest(
        owner_id="user_123",
        display_name="Primary Voice",
        consent_confirmed=True,
        ownership_confirmed=False,  # Violated
        purposes=[ConsentPurpose.STORE_VOICE_REFERENCE],
    )

    with pytest.raises(InvalidConsentError) as exc_info:
        service.enroll_voice(synthetic_speech_wav, req)
    assert "ownership_confirmed=true" in str(exc_info.value)


def test_consent_missing_purposes(tmp_path: Path, synthetic_speech_wav: Path):
    storage = VoiceProfileStorage(base_dir=tmp_path / "voices")
    service = SpeakerProfileService(storage=storage)

    req = VoiceEnrollmentRequest(
        owner_id="user_123",
        display_name="Primary Voice",
        consent_confirmed=True,
        ownership_confirmed=True,
        purposes=[],  # Violated
    )

    with pytest.raises(InvalidConsentError) as exc_info:
        service.enroll_voice(synthetic_speech_wav, req)
    assert "At least one specific processing purpose" in str(exc_info.value)


def test_consent_missing_store_purpose(tmp_path: Path, synthetic_speech_wav: Path):
    storage = VoiceProfileStorage(base_dir=tmp_path / "voices")
    service = SpeakerProfileService(storage=storage)

    req = VoiceEnrollmentRequest(
        owner_id="user_123",
        display_name="Primary Voice",
        consent_confirmed=True,
        ownership_confirmed=True,
        purposes=[ConsentPurpose.GENERATE_SPEECH],  # Missing STORE_VOICE_REFERENCE
    )

    with pytest.raises(InvalidConsentError) as exc_info:
        service.enroll_voice(synthetic_speech_wav, req)
    assert "store_voice_reference" in str(exc_info.value)


# ── Audio Quality & Acoustic Rejection Tests ────────────────────────────────


def test_validator_rejects_silence(synthetic_silence_wav: Path):
    validator = EnrollmentValidator()
    with pytest.raises(EnrollmentValidationError) as exc_info:
        validator.validate_enrollment_audio(synthetic_silence_wav)
    assert any("silence" in issue.lower() for issue in exc_info.value.issues)


def test_validator_rejects_clipped_audio(synthetic_clipped_wav: Path):
    validator = EnrollmentValidator()
    with pytest.raises(EnrollmentValidationError) as exc_info:
        validator.validate_enrollment_audio(synthetic_clipped_wav)
    assert any("clipping" in issue.lower() for issue in exc_info.value.issues)


def test_validator_rejects_too_short_audio(synthetic_short_wav: Path):
    validator = EnrollmentValidator()
    with pytest.raises(EnrollmentValidationError) as exc_info:
        validator.validate_enrollment_audio(synthetic_short_wav, min_duration=3.0)
    assert any("below minimum enrollment requirement" in issue.lower() for issue in exc_info.value.issues)


def test_validator_rejects_pure_tone_without_speech(tmp_path: Path):
    """Silero VAD classifies a flat 440 Hz pure tone as non-speech."""
    wav_path = tmp_path / "tone.wav"
    sr = 16000
    duration = 4.0
    tone = (0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32))).astype(np.float32)
    sf.write(str(wav_path), tone, sr, subtype="PCM_16")

    validator = EnrollmentValidator()
    with pytest.raises(EnrollmentValidationError) as exc_info:
        validator.validate_enrollment_audio(wav_path, min_speech_duration=1.5)
    assert any("speech" in issue.lower() for issue in exc_info.value.issues)
