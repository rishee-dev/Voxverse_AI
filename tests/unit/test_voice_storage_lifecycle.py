"""Unit tests for VoiceProfileStorage lifecycle, security controls, and deletion verification.

Owner: P1 — Audio and Speaker Profile Engineer
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pytest

from voxverse.genai.speaker.exceptions import (
    PathTraversalError,
    ProfileNotFoundError,
    UnauthorizedProfileAccessError,
)
from voxverse.genai.speaker.schemas import (
    ConsentPurpose,
    ConsentRecord,
    ConsentStatus,
    EnrollmentStatus,
    EnrollmentValidationReport,
    VoiceProfile,
)
from voxverse.storage.voice_storage import VoiceProfileStorage


@pytest.fixture
def mock_profile() -> VoiceProfile:
    now = datetime.now(timezone.utc)
    consent = ConsentRecord(
        speaker_id="spk_test123",
        owner_id="user_alice",
        status=ConsentStatus.GRANTED,
        granted_at=now,
        consent_version="1.0",
        purposes=[ConsentPurpose.STORE_VOICE_REFERENCE, ConsentPurpose.GENERATE_SPEECH],
        recording_ownership_confirmed=True,
    )
    val_report = EnrollmentValidationReport(
        is_valid=True,
        duration_seconds=5.0,
        sample_rate=16000,
        channels=1,
        rms_amplitude=0.15,
        peak_amplitude=0.75,
        is_silent=False,
        is_clipped=False,
        detected_speech_duration_seconds=3.5,
        speech_ratio=0.7,
    )
    return VoiceProfile(
        profile_id="spk_test123",
        owner_id="user_alice",
        display_name="Alice Voice",
        created_at=now,
        updated_at=now,
        consent=consent,
        validation_report=val_report,
    )


def test_storage_save_and_retrieve(tmp_path: Path, mock_profile: VoiceProfile):
    storage = VoiceProfileStorage(base_dir=tmp_path / "voices")
    dummy_audio = np.zeros(16000 * 5, dtype=np.float32)

    audio_dest = storage.save_profile(mock_profile, dummy_audio, sample_rate=16000)
    assert audio_dest.exists()

    retrieved = storage.get_profile("spk_test123", owner_id="user_alice")
    assert retrieved.profile_id == "spk_test123"
    assert retrieved.owner_id == "user_alice"
    assert retrieved.display_name == "Alice Voice"

    # Verify safe public metadata omits internal filesystem paths
    meta = retrieved.to_metadata()
    meta_dict = meta.model_dump()
    assert "reference_audio_filename" not in meta_dict
    assert "audio_path" not in meta_dict


def test_storage_ownership_enforcement(tmp_path: Path, mock_profile: VoiceProfile):
    storage = VoiceProfileStorage(base_dir=tmp_path / "voices")
    dummy_audio = np.zeros(16000 * 5, dtype=np.float32)
    storage.save_profile(mock_profile, dummy_audio, sample_rate=16000)

    # User Bob attempts to access User Alice's profile
    with pytest.raises(UnauthorizedProfileAccessError) as exc_info:
        storage.get_profile("spk_test123", owner_id="user_bob")
    assert "authorization" in str(exc_info.value).lower()
    assert exc_info.value.code == "UNAUTHORIZED_PROFILE_ACCESS"


def test_storage_list_profiles_owner_scoped(tmp_path: Path, mock_profile: VoiceProfile):
    storage = VoiceProfileStorage(base_dir=tmp_path / "voices")
    dummy_audio = np.zeros(16000 * 5, dtype=np.float32)
    storage.save_profile(mock_profile, dummy_audio, sample_rate=16000)

    # User Alice sees her profile
    alice_profiles = storage.list_profiles("user_alice")
    assert len(alice_profiles) == 1
    assert alice_profiles[0].profile_id == "spk_test123"

    # User Bob sees zero profiles
    bob_profiles = storage.list_profiles("user_bob")
    assert len(bob_profiles) == 0


def test_storage_path_traversal_prevention(tmp_path: Path):
    storage = VoiceProfileStorage(base_dir=tmp_path / "voices")

    # Injection attempts with directory traversal characters
    traversal_ids = [
        "../escape",
        "..\\escape",
        "profile/sub",
        "../../etc/passwd",
        "spk;rm -rf",
        "spk with spaces",
    ]
    for bad_id in traversal_ids:
        with pytest.raises(PathTraversalError):
            storage.get_profile(bad_id, owner_id="user_alice")

        with pytest.raises(PathTraversalError):
            storage.get_profile("spk_valid", owner_id=bad_id)


def test_storage_deletion_and_verification(tmp_path: Path, mock_profile: VoiceProfile):
    storage = VoiceProfileStorage(base_dir=tmp_path / "voices")
    dummy_audio = np.zeros(16000 * 5, dtype=np.float32)
    storage.save_profile(mock_profile, dummy_audio, sample_rate=16000)

    profile_dir = tmp_path / "voices" / "spk_test123"
    audio_file = profile_dir / "reference.wav"
    profile_file = profile_dir / "profile.json"
    assert audio_file.exists()
    assert profile_file.exists()

    # Execute deletion
    result = storage.delete_profile("spk_test123", owner_id="user_alice")

    assert result.all_verified_deleted is True
    assert result.reference_audio_deleted is True
    assert result.metadata_deleted is True

    # Verify on physical disk that files are GONE
    assert not audio_file.exists()
    assert not profile_file.exists()

    # Verify audit tombstone exists without biometric audio data
    tombstone = profile_dir / "tombstone.json"
    assert tombstone.exists()
    data = json.loads(tombstone.read_text(encoding="utf-8"))
    assert data["profile_id"] == "spk_test123"
    assert data["consent_status"] == "revoked"
    assert data["biometric_audio_deleted"] is True
    assert "audio_bytes" not in data

    # Attempting to retrieve deleted profile raises ProfileNotFoundError
    with pytest.raises(ProfileNotFoundError):
        storage.get_profile("spk_test123", owner_id="user_alice")
