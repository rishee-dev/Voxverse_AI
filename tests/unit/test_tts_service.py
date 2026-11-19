"""Unit tests for TTS service, model loader, validators, and runtime safeguards.

Owner: P3 — Voice Generation Engineer
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from voxverse.genai.speaker.schemas import (
    ConsentPurpose,
    ConsentRecord,
    ConsentStatus,
    EnrollmentStatus,
    EnrollmentValidationReport,
    VoiceProfile,
)
from voxverse.genai.tts.benchmarks import get_benchmark_suite
from voxverse.genai.tts.config import TTSConfig
from voxverse.genai.tts.exceptions import (
    ConcurrentRequestError,
    GatedRepositoryError,
    InferenceTimeoutError,
    InvalidAudioOutputError,
    MemoryAdmissionError,
    MissingCheckpointError,
    MissingConsentError,
    MissingReferenceTranscriptError,
    MissingVocoderError,
    ModelLoadError,
    ReferenceAudioError,
    TeluguScriptValidationError,
    VoiceProfileNotFoundError,
)
from voxverse.genai.tts.indicf5_engine import IndicF5Engine
from voxverse.genai.tts.model_loader import TTSModelLoader
from voxverse.genai.tts.schemas import (
    SpeakerConditioningData,
    TTSRequest,
)
from voxverse.genai.tts.service import TTSService
from voxverse.genai.tts.validator import (
    validate_audio_output,
    validate_reference_audio,
    validate_telugu_text,
)


@pytest.fixture
def sample_wav_path(tmp_path: Path) -> Path:
    """Create a temporary valid 3-second 16kHz mono WAV file."""
    wav_path = tmp_path / "reference.wav"
    sr = 16000
    samples = np.zeros(sr * 3, dtype=np.float32)
    sf.write(str(wav_path), samples, sr)
    return wav_path


@pytest.fixture
def mock_voice_storage(sample_wav_path: Path) -> MagicMock:
    """Mock VoiceProfileStorage with valid enrolled profile and consent."""
    storage = MagicMock()
    now = datetime.now(timezone.utc)
    profile = VoiceProfile(
        profile_id="voice-test-01",
        owner_id="owner-42",
        display_name="Kalyan",
        created_at=now,
        updated_at=now,
        consent=ConsentRecord(
            speaker_id="voice-test-01",
            owner_id="owner-42",
            status=ConsentStatus.GRANTED,
            purposes=[ConsentPurpose.STORE_VOICE_REFERENCE, ConsentPurpose.GENERATE_SPEECH],
            recording_ownership_confirmed=True,
        ),
        validation_report=EnrollmentValidationReport(
            is_valid=True,
            duration_seconds=3.0,
            sample_rate=16000,
            channels=1,
            rms_amplitude=0.15,
            peak_amplitude=0.6,
            is_silent=False,
            is_clipped=False,
            detected_speech_duration_seconds=2.8,
            speech_ratio=0.93,
        ),
        reference_audio_filename="reference.wav",
        enrollment_status=EnrollmentStatus.ENROLLED,
    )
    storage.get_profile.return_value = profile
    storage.get_reference_audio_path.return_value = sample_wav_path
    return storage


def test_tts_config_defaults() -> None:
    config = TTSConfig()
    assert config.model_id == "ai4bharat/IndicF5"
    assert config.vocoder_id == "charactr/vocos-mel-24khz"
    assert config.device == "cpu"
    assert config.nfe_steps == 16
    assert config.memory_min_available_gb == 4.0
    assert config.inference_timeout_seconds == 60.0


def test_memory_admission_policy_tiers() -> None:
    config = TTSConfig(memory_min_available_gb=4.0, memory_safe_available_gb=5.0)

    # Below 4 GB: Blocked
    with patch("psutil.virtual_memory") as mock_vm:
        mock_vm.return_value = MagicMock(available=3.5 * (1024**3), total=16.0 * (1024**3), percent=78.1)
        status = config.check_memory_admission()
        assert not status.admitted
        assert status.tier == "TIER_1_BLOCKED"

    # Between 4 and 5 GB: Restricted
    with patch("psutil.virtual_memory") as mock_vm:
        mock_vm.return_value = MagicMock(available=4.5 * (1024**3), total=16.0 * (1024**3), percent=71.8)
        status = config.check_memory_admission()
        assert not status.admitted
        assert status.tier == "TIER_2_RESTRICTED"

    # At or above 5 GB: Admitted
    with patch("psutil.virtual_memory") as mock_vm:
        mock_vm.return_value = MagicMock(available=5.5 * (1024**3), total=16.0 * (1024**3), percent=65.6)
        status = config.check_memory_admission()
        assert status.admitted
        assert status.tier == "TIER_3_ADMITTED"


def test_model_loader_memory_admission_rejection() -> None:
    config = TTSConfig(memory_min_available_gb=4.0)
    loader = TTSModelLoader(config)

    with patch("psutil.virtual_memory") as mock_vm:
        mock_vm.return_value = MagicMock(available=2.5 * (1024**3), total=16.0 * (1024**3), percent=84.3)
        with pytest.raises(MemoryAdmissionError) as exc:
            loader.check_memory_admission(strict=True)
        assert exc.value.available_gb == 2.5
        assert exc.value.required_gb == 4.0


def test_model_loader_missing_checkpoint(tmp_path: Path) -> None:
    config = TTSConfig(model_cache_dir=str(tmp_path / "empty_cache"))
    loader = TTSModelLoader(config)

    with patch("huggingface_hub.hf_hub_download", side_effect=FileNotFoundError("Not found")):
        with pytest.raises(MissingCheckpointError):
            loader.resolve_checkpoint_path()


def test_model_loader_gated_repository_error() -> None:
    from huggingface_hub.errors import GatedRepoError
    config = TTSConfig()
    loader = TTSModelLoader(config)

    with patch("huggingface_hub.hf_hub_download", side_effect=GatedRepoError("401 Gated")):
        with pytest.raises(GatedRepositoryError) as exc:
            loader.resolve_checkpoint_path()
        assert "ai4bharat/IndicF5" in exc.value.repo_id


def test_model_loader_missing_vocoder() -> None:
    config = TTSConfig()
    loader = TTSModelLoader(config)

    with patch("huggingface_hub.hf_hub_download", side_effect=Exception("Network error")):
        with pytest.raises(MissingVocoderError):
            loader.resolve_vocoder_path("invalid/vocoder-path")


def test_validator_telugu_text() -> None:
    # Valid Telugu text
    valid = validate_telugu_text("నమస్కారం ప్రపంచం")
    assert valid == "నమస్కారం ప్రపంచం"

    # Empty text
    with pytest.raises(TeluguScriptValidationError) as exc:
        validate_telugu_text("")
    assert "empty" in str(exc.value)

    # Non-Telugu text
    with pytest.raises(TeluguScriptValidationError) as exc:
        validate_telugu_text("Hello World! 12345")
    assert "Unicode Telugu script block" in str(exc.value)

    # Exceeding length
    with pytest.raises(TeluguScriptValidationError) as exc:
        validate_telugu_text("న" * 501, max_characters=500)
    assert "exceeds maximum" in str(exc.value)


def test_validator_reference_audio(sample_wav_path: Path, tmp_path: Path) -> None:
    # Valid audio
    info = validate_reference_audio(str(sample_wav_path), min_duration=1.0, max_duration=10.0)
    assert info.samplerate == 16000

    # Non-existent file
    with pytest.raises(ReferenceAudioError):
        validate_reference_audio(str(tmp_path / "non_existent.wav"))

    # Too short audio
    with pytest.raises(ReferenceAudioError) as exc:
        validate_reference_audio(str(sample_wav_path), min_duration=5.0)
    assert "shorter than minimum" in str(exc.value)

    # Too long audio
    with pytest.raises(ReferenceAudioError) as exc:
        validate_reference_audio(str(sample_wav_path), max_duration=1.0)
    assert "exceeds maximum" in str(exc.value)


def test_validator_audio_output() -> None:
    # Valid audio
    samples = np.array([0.1, 0.2, -0.1, -0.2] * 2400, dtype=np.float32)
    validate_audio_output(samples, 24000)

    # Empty audio
    with pytest.raises(InvalidAudioOutputError):
        validate_audio_output(np.array([], dtype=np.float32), 24000)

    # NaN / Inf audio
    with pytest.raises(InvalidAudioOutputError):
        validate_audio_output(np.array([0.1, np.nan, 0.2]), 24000)

    # Audio too short (<0.1s)
    with pytest.raises(InvalidAudioOutputError):
        validate_audio_output(np.array([0.1, 0.2]), 24000)


def test_indicf5_engine_synthesize_success(sample_wav_path: Path, tmp_path: Path) -> None:
    config = TTSConfig(model_cache_dir=str(tmp_path))
    engine = IndicF5Engine(config)
    engine._loaded = True  # Mock loaded state

    out_file = tmp_path / "output.wav"
    cond = SpeakerConditioningData(
        voice_id="voice-01",
        owner_id="user-1",
        speaker_name="Kalyan",
        reference_audio_path=str(sample_wav_path),
        reference_transcript="This is my verified voice.",
        consent_hash="hash123",
    )
    result = engine.synthesize(
        text="నమస్కారం",
        conditioning_data=cond,
        output_path=str(out_file),
    )
    assert result.audio_path == str(out_file)
    assert out_file.exists()
    assert result.audio_duration_seconds > 0.0
    assert result.real_time_factor >= 0.0
    assert result.speaker_similarity_score == 0.72


def test_indicf5_engine_concurrent_request_rejection(tmp_path: Path) -> None:
    config = TTSConfig()
    engine = IndicF5Engine(config)
    engine._loaded = True

    # Lock the engine artificially to simulate an active inference
    engine._lock.acquire()
    try:
        with pytest.raises(ConcurrentRequestError):
            engine.synthesize("నమస్కారం", output_path=str(tmp_path / "out.wav"))
    finally:
        engine._lock.release()


def test_indicf5_engine_timeout(tmp_path: Path) -> None:
    config = TTSConfig(inference_timeout_seconds=5.0)
    engine = IndicF5Engine(config)
    engine._loaded = True

    # Artificially delay inside timeout check by having perf_counter return elapsed > 5.0
    with patch("time.perf_counter", side_effect=[0.0, 10.0, 20.0, 30.0]):
        with pytest.raises(InferenceTimeoutError):
            engine.synthesize("నమస్కారం", output_path=str(tmp_path / "out.wav"))


def test_tts_service_orchestration(mock_voice_storage: MagicMock, tmp_path: Path) -> None:
    config = TTSConfig()
    engine = IndicF5Engine(config)
    engine._loaded = True
    service = TTSService(engine=engine, voice_storage=mock_voice_storage)

    req = TTSRequest(
        text="నమస్కారం, ఇది వోక్స్ వర్స్ పరీక్ష.",
        voice_id="voice-test-01",
    )
    out_file = tmp_path / "out_service.wav"
    result = service.synthesize(
        req,
        reference_transcript="Hello, welcome to VoxVerse audio systems.",
        output_path=str(out_file),
    )
    assert result.audio_path == str(out_file)
    assert result.speaker_similarity_score is not None


def test_tts_service_missing_consent(mock_voice_storage: MagicMock) -> None:
    mock_voice_storage.get_profile.return_value.consent.status = ConsentStatus.REVOKED
    engine = IndicF5Engine()
    service = TTSService(engine=engine, voice_storage=mock_voice_storage)

    req = TTSRequest(text="నమస్కారం", voice_id="voice-test-01")
    with pytest.raises(MissingConsentError):
        service.synthesize(req, reference_transcript="Hello world")


def test_tts_service_missing_transcript(mock_voice_storage: MagicMock) -> None:
    engine = IndicF5Engine()
    service = TTSService(engine=engine, voice_storage=mock_voice_storage)

    req = TTSRequest(text="నమస్కారం", voice_id="voice-test-01")
    with pytest.raises(MissingReferenceTranscriptError):
        service.synthesize(req, reference_transcript="")


def test_tts_service_profile_not_found(mock_voice_storage: MagicMock) -> None:
    mock_voice_storage.get_profile.return_value = None
    engine = IndicF5Engine()
    service = TTSService(engine=engine, voice_storage=mock_voice_storage)

    req = TTSRequest(text="నమస్కారం", voice_id="unknown-id")
    with pytest.raises(VoiceProfileNotFoundError):
        service.synthesize(req, reference_transcript="Hello")


def test_benchmark_suite_validity() -> None:
    benchmarks = get_benchmark_suite()
    assert len(benchmarks) == 10
    ids = [b.id for b in benchmarks]
    assert ids == [f"TTS-{i:02d}" for i in range(1, 11)]
    for b in benchmarks:
        assert b.telugu_text
        assert b.english_translation
        assert b.phonetic_focus
        # Validate that each benchmark text passes Telugu validation
        validated = validate_telugu_text(b.telugu_text)
        assert len(validated) > 0
