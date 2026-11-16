"""Unit tests for the GET /api/v1/info endpoint.

Covers:
- HTTP 200 OK status code
- Presence of required response fields
- Supported source and target languages (English -> Telugu)
- Pipeline feature implementation statuses (all 'not_implemented' for Phase 0)
- Processing mode ('batch')
- Team ownership and role allocation (P1 through P4)
- Version and environment alignment with configuration
- Correlation headers and HTTP method enforcement

Owner: P4 (Frontend & Integration Engineer) / Shared
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from voxverse.api.app import app
from voxverse.api.schemas import FeatureStatus
from voxverse.config.settings import get_settings


@pytest.mark.asyncio
async def test_info_returns_status_200(async_client: AsyncClient) -> None:
    """Test that GET /api/v1/info returns HTTP 200 OK.

    Arrange: Async HTTP client connected to the application.
    Act: Send GET request to /api/v1/info.
    Assert: HTTP status code is 200.
    """
    # Arrange
    endpoint = "/api/v1/info"

    # Act
    response = await async_client.get(endpoint)

    # Assert
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_info_response_has_required_fields(async_client: AsyncClient) -> None:
    """Test that GET /api/v1/info contains all required schema fields.

    Arrange: Async HTTP client connected to the application.
    Act: Send GET request to /api/v1/info.
    Assert: JSON body contains all required InfoResponse fields.
    """
    # Arrange
    required_fields = {
        "version",
        "environment",
        "supported_source_languages",
        "supported_target_languages",
        "processing_mode",
        "features",
        "team",
    }

    # Act
    response = await async_client.get("/api/v1/info")
    payload = response.json()

    # Assert
    assert required_fields.issubset(payload.keys())


@pytest.mark.asyncio
async def test_info_supported_source_languages(async_client: AsyncClient) -> None:
    """Test that supported source languages contains English ('en').

    Arrange: Async HTTP client connected to the application.
    Act: Send GET request to /api/v1/info and extract supported_source_languages.
    Assert: 'en' is in supported_source_languages.
    """
    # Arrange & Act
    response = await async_client.get("/api/v1/info")
    sources = response.json()["supported_source_languages"]

    # Assert
    assert sources == ["en"]


@pytest.mark.asyncio
async def test_info_supported_target_languages(async_client: AsyncClient) -> None:
    """Test that supported target languages contains Telugu ('te').

    Arrange: Async HTTP client connected to the application.
    Act: Send GET request to /api/v1/info and extract supported_target_languages.
    Assert: 'te' is in supported_target_languages.
    """
    # Arrange & Act
    response = await async_client.get("/api/v1/info")
    targets = response.json()["supported_target_languages"]

    # Assert
    assert targets == ["te"]


@pytest.mark.asyncio
async def test_info_processing_mode_is_batch(async_client: AsyncClient) -> None:
    """Test that processing_mode is 'batch' in MVP Phase 0.

    Arrange: Async HTTP client connected to the application.
    Act: Send GET request to /api/v1/info.
    Assert: 'processing_mode' equals 'batch'.
    """
    # Arrange
    expected_mode = "batch"

    # Act
    response = await async_client.get("/api/v1/info")
    data = response.json()

    # Assert
    assert data["processing_mode"] == expected_mode


@pytest.mark.asyncio
async def test_info_all_features_show_not_implemented(async_client: AsyncClient) -> None:
    """Test that all pipeline features currently show 'not_implemented' status.

    Arrange: Async HTTP client connected to the application.
    Act: Send GET request to /api/v1/info and retrieve features list.
    Assert: Every feature has status == 'not_implemented'.
    """
    # Arrange
    expected_status = FeatureStatus.NOT_IMPLEMENTED.value

    # Act
    response = await async_client.get("/api/v1/info")
    features = response.json()["features"]

    # Assert
    assert len(features) > 0
    for feature in features:
        assert feature["status"] == expected_status, (
            f"Feature '{feature['name']}' expected status '{expected_status}', got '{feature['status']}'"
        )


@pytest.mark.asyncio
async def test_info_contains_expected_feature_names(async_client: AsyncClient) -> None:
    """Test that info reports all core Phase 0 pipeline features by name.

    Arrange: Expected list of core features for VoxVerse pipeline.
    Act: Send GET request to /api/v1/info.
    Assert: Feature names set matches {audio_upload, transcription, translation, voice_cloning, tts}.
    """
    # Arrange
    expected_feature_names = {
        "audio_upload",
        "transcription",
        "translation",
        "voice_cloning",
        "tts",
    }

    # Act
    response = await async_client.get("/api/v1/info")
    features = response.json()["features"]
    feature_names = {f["name"] for f in features}

    # Assert
    assert feature_names == expected_feature_names


@pytest.mark.asyncio
async def test_info_team_members_ownership(async_client: AsyncClient) -> None:
    """Test that all four project team roles (P1-P4) are represented.

    Arrange: Expected team member IDs P1, P2, P3, P4.
    Act: Send GET request to /api/v1/info and extract team member IDs.
    Assert: The team IDs set exactly matches {'P1', 'P2', 'P3', 'P4'}.
    """
    # Arrange
    expected_team_ids = {"P1", "P2", "P3", "P4"}

    # Act
    response = await async_client.get("/api/v1/info")
    team = response.json()["team"]
    team_ids = {member["id"] for member in team}

    # Assert
    assert team_ids == expected_team_ids


@pytest.mark.asyncio
async def test_info_version_matches_settings(async_client: AsyncClient) -> None:
    """Test that info endpoint version matches application settings version.

    Arrange: Read current app_version from settings.
    Act: Send GET request to /api/v1/info.
    Assert: Response version matches settings version.
    """
    # Arrange
    settings = get_settings()

    # Act
    response = await async_client.get("/api/v1/info")
    data = response.json()

    # Assert
    assert data["version"] == settings.app_version


@pytest.mark.asyncio
async def test_info_environment_matches_settings(async_client: AsyncClient) -> None:
    """Test that info endpoint environment matches configured environment.

    Arrange: Read current environment value from settings.
    Act: Send GET request to /api/v1/info.
    Assert: Response environment matches settings environment value.
    """
    # Arrange
    settings = get_settings()

    # Act
    response = await async_client.get("/api/v1/info")
    data = response.json()

    # Assert
    assert data["environment"] == settings.environment.value


@pytest.mark.asyncio
async def test_info_method_not_allowed_for_post(async_client: AsyncClient) -> None:
    """Test that POST /api/v1/info is rejected with HTTP 405 Method Not Allowed.

    Arrange: Async HTTP client connected to the application.
    Act: Send POST request to /api/v1/info.
    Assert: HTTP status code is 405.
    """
    # Arrange & Act
    response = await async_client.post("/api/v1/info", json={})

    # Assert
    assert response.status_code == 405
