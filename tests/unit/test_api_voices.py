"""Unit tests for FastAPI voice profile endpoints (/api/v1/voices).

Owner: P4 — Pipeline Engineer & P1 — Audio/Speaker Engineer
"""

from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from voxverse.api.app import create_app
from voxverse.config.settings import Settings, get_settings


@pytest.fixture
def test_app(tmp_path: Path):
    """Create test application with an isolated storage directory."""
    settings = Settings(
        environment="testing",
        debug=True,
        storage_dir=str(tmp_path / "storage"),
        voices_dir="voices",
        upload_dir=str(tmp_path / "uploads"),
        temp_dir=str(tmp_path / "temp"),
    )
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    return app


@pytest.fixture
def client(test_app) -> TestClient:
    return TestClient(test_app)


@pytest.fixture
def sample_audio_bytes() -> bytes:
    """Provide verified LibriSpeech 3.5s speech audio bytes."""
    librispeech_sample = Path("data/benchmark/librispeech/wav_16k/6930-75918-0000.wav")
    if librispeech_sample.exists():
        return librispeech_sample.read_bytes()

    buf = io.BytesIO()
    sr = 16000
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    audio = (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def test_api_missing_auth_header(client: TestClient, sample_audio_bytes: bytes):
    """Requesting enrollment without X-User-ID header is rejected with 401."""
    response = client.post(
        "/api/v1/voices",
        files={"file": ("enroll.wav", sample_audio_bytes, "audio/wav")},
        data={
            "display_name": "Test Voice",
            "consent_confirmed": "true",
            "ownership_confirmed": "true",
            "purposes": "store_voice_reference,generate_speech",
        },
    )
    assert response.status_code == 401
    assert "Authentication required" in response.json()["detail"]


def test_api_invalid_user_id_traversal(client: TestClient, sample_audio_bytes: bytes):
    """User ID with path traversal characters is rejected with 400."""
    response = client.post(
        "/api/v1/voices",
        headers={"X-User-ID": "../malicious_user"},
        files={"file": ("enroll.wav", sample_audio_bytes, "audio/wav")},
        data={
            "display_name": "Test Voice",
            "consent_confirmed": "true",
            "ownership_confirmed": "true",
            "purposes": "store_voice_reference,generate_speech",
        },
    )
    assert response.status_code == 400
    assert "Invalid caller identity" in response.json()["detail"]


def test_api_enroll_without_consent(client: TestClient, sample_audio_bytes: bytes):
    """Enrollment with consent_confirmed=false is rejected with 400."""
    response = client.post(
        "/api/v1/voices",
        headers={"X-User-ID": "user_alice"},
        files={"file": ("enroll.wav", sample_audio_bytes, "audio/wav")},
        data={
            "display_name": "Alice Voice",
            "consent_confirmed": "false",  # Violated
            "ownership_confirmed": "true",
            "purposes": "store_voice_reference,generate_speech",
        },
    )
    assert response.status_code == 400
    assert "affirmative consent" in response.json()["detail"].lower()


def test_api_successful_lifecycle(client: TestClient, sample_audio_bytes: bytes):
    """End-to-end API test: enroll -> get -> list -> delete -> verify absent."""
    headers_alice = {"X-User-ID": "user_alice"}
    headers_bob = {"X-User-ID": "user_bob"}

    # 1. Enroll voice
    enroll_resp = client.post(
        "/api/v1/voices",
        headers=headers_alice,
        files={"file": ("enroll.wav", sample_audio_bytes, "audio/wav")},
        data={
            "display_name": "Alice English Voice",
            "consent_confirmed": "true",
            "ownership_confirmed": "true",
            "purposes": "store_voice_reference,generate_speech",
            "consent_version": "1.0",
        },
    )
    assert enroll_resp.status_code == 201
    profile = enroll_resp.json()
    voice_id = profile["profile_id"]
    assert voice_id.startswith("spk_")
    assert profile["owner_id"] == "user_alice"
    assert profile["display_name"] == "Alice English Voice"
    assert profile["consent_status"] == "granted"

    # Verify no internal filesystem path leakage in JSON response
    resp_text = enroll_resp.text
    assert "reference_audio_filename" not in resp_text
    assert "reference.wav" not in resp_text

    # 2. Get profile as owner
    get_resp = client.get(f"/api/v1/voices/{voice_id}", headers=headers_alice)
    assert get_resp.status_code == 200
    assert get_resp.json()["profile_id"] == voice_id

    # 3. Get profile as unauthorized user Bob -> 403 Forbidden
    get_bob_resp = client.get(f"/api/v1/voices/{voice_id}", headers=headers_bob)
    assert get_bob_resp.status_code == 403
    assert "authorization" in get_bob_resp.json()["detail"].lower()

    # 4. List profiles
    list_alice = client.get("/api/v1/voices", headers=headers_alice)
    assert list_alice.status_code == 200
    assert len(list_alice.json()) == 1
    assert list_alice.json()[0]["profile_id"] == voice_id

    list_bob = client.get("/api/v1/voices", headers=headers_bob)
    assert list_bob.status_code == 200
    assert len(list_bob.json()) == 0

    # 5. Delete profile as unauthorized user Bob -> 403 Forbidden
    del_bob = client.delete(f"/api/v1/voices/{voice_id}", headers=headers_bob)
    assert del_bob.status_code == 403

    # 6. Delete profile as owner Alice -> 200 OK
    del_alice = client.delete(f"/api/v1/voices/{voice_id}", headers=headers_alice)
    assert del_alice.status_code == 200
    del_result = del_alice.json()
    assert del_result["all_verified_deleted"] is True
    assert del_result["reference_audio_deleted"] is True

    # 7. Post-deletion access returns 404 Not Found
    get_post_del = client.get(f"/api/v1/voices/{voice_id}", headers=headers_alice)
    assert get_post_del.status_code == 404
