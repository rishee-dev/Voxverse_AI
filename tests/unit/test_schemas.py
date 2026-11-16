"""Unit tests for VoxVerse Pydantic API schemas.

Covers:
- HealthResponse validation, serialization, and deserialization
- InfoResponse validation and nested model serialization (FeatureInfo, TeamMember)
- ErrorResponse and ErrorDetail validation, serialization, and default values
- FeatureStatus enumeration values and validation
- Schema rejection of invalid or missing required fields

Owner: Shared (all team members)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from voxverse.api.schemas import (
    ErrorDetail,
    ErrorResponse,
    FeatureInfo,
    FeatureStatus,
    HealthResponse,
    InfoResponse,
    TeamMember,
)


# ── HealthResponse Tests ────────────────────────────────────────────────


def test_health_response_serialization_to_dict() -> None:
    """Test serializing HealthResponse to dictionary via model_dump().

    Arrange: HealthResponse instance with known values.
    Act: Dump model to dictionary.
    Assert: Dictionary contains expected keys and matching values.
    """
    # Arrange
    now = datetime(2026, 11, 16, 12, 0, 0, tzinfo=timezone.utc)
    health = HealthResponse(status="healthy", version="0.1.0", timestamp=now)

    # Act
    data = health.model_dump()

    # Assert
    assert data["status"] == "healthy"
    assert data["version"] == "0.1.0"
    assert data["timestamp"] == now


def test_health_response_serialization_to_json() -> None:
    """Test serializing HealthResponse to JSON string via model_dump_json().

    Arrange: HealthResponse instance with known timestamp.
    Act: Dump model to JSON string and parse back with json.loads.
    Assert: Parsed JSON reflects expected string format and ISO timestamp.
    """
    # Arrange
    now = datetime(2026, 11, 16, 12, 0, 0, tzinfo=timezone.utc)
    health = HealthResponse(status="healthy", version="0.1.0", timestamp=now)

    # Act
    json_str = health.model_dump_json()
    parsed = json.loads(json_str)

    # Assert
    assert parsed["status"] == "healthy"
    assert parsed["version"] == "0.1.0"
    assert "2026-11-16T12:00:00" in parsed["timestamp"]


def test_health_response_deserialization_from_dict() -> None:
    """Test deserializing dictionary to HealthResponse model.

    Arrange: Raw payload with ISO timestamp string.
    Act: Validate payload into HealthResponse model.
    Assert: Attributes match and timestamp is parsed to a datetime object.
    """
    # Arrange
    raw_payload = {
        "status": "healthy",
        "version": "0.1.0",
        "timestamp": "2026-11-16T12:00:00Z",
    }

    # Act
    model = HealthResponse.model_validate(raw_payload)

    # Assert
    assert model.status == "healthy"
    assert model.version == "0.1.0"
    assert isinstance(model.timestamp, datetime)


def test_health_response_missing_required_fields_raises_validation_error() -> None:
    """Test that creating HealthResponse without required fields raises ValidationError.

    Arrange: Missing 'status' and 'timestamp' fields.
    Act & Assert: Instantiation raises ValidationError referencing missing fields.
    """
    # Arrange
    incomplete_data = {"version": "0.1.0"}

    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        HealthResponse.model_validate(incomplete_data)
    missing_locs = {err["loc"][0] for err in exc_info.value.errors()}
    assert "status" in missing_locs
    assert "timestamp" in missing_locs


# ── FeatureStatus and FeatureInfo Tests ─────────────────────────────────


def test_feature_status_enum_values() -> None:
    """Test that FeatureStatus defines expected enum states.

    Arrange: Expected string values.
    Act: Retrieve FeatureStatus members.
    Assert: NOT_IMPLEMENTED, EXPERIMENTAL, and STABLE exist with correct values.
    """
    # Arrange & Act & Assert
    assert FeatureStatus.NOT_IMPLEMENTED.value == "not_implemented"
    assert FeatureStatus.EXPERIMENTAL.value == "experimental"
    assert FeatureStatus.STABLE.value == "stable"


def test_feature_info_serialization() -> None:
    """Test FeatureInfo model serialization to dictionary.

    Arrange: FeatureInfo instance.
    Act: Dump model to dict.
    Assert: Fields match input data.
    """
    # Arrange
    feature = FeatureInfo(
        name="audio_upload",
        status=FeatureStatus.NOT_IMPLEMENTED,
        description="Upload audio files",
    )

    # Act
    data = feature.model_dump()

    # Assert
    assert data["name"] == "audio_upload"
    assert data["status"] == "not_implemented"
    assert data["description"] == "Upload audio files"


def test_feature_info_default_description() -> None:
    """Test that FeatureInfo provides an empty string default for description.

    Arrange: Instantiate FeatureInfo without explicit description.
    Act: Check description attribute.
    Assert: Defaults to empty string.
    """
    # Arrange & Act
    feature = FeatureInfo(name="transcription", status=FeatureStatus.NOT_IMPLEMENTED)

    # Assert
    assert feature.description == ""


def test_feature_info_invalid_status_raises_error() -> None:
    """Test that invalid FeatureStatus value raises ValidationError.

    Arrange: Invalid status string 'completed'.
    Act & Assert: Validation raises ValidationError.
    """
    # Arrange
    invalid_data = {"name": "tts", "status": "completed"}

    # Act & Assert
    with pytest.raises(ValidationError):
        FeatureInfo.model_validate(invalid_data)


# ── TeamMember Tests ────────────────────────────────────────────────────


def test_team_member_serialization() -> None:
    """Test TeamMember model serialization and attributes.

    Arrange: TeamMember instance for P1.
    Act: Dump model to dict.
    Assert: Dictionary contains expected 'id' and 'role'.
    """
    # Arrange
    member = TeamMember(id="P1", role="Project Lead & Backend Architect")

    # Act
    data = member.model_dump()

    # Assert
    assert data["id"] == "P1"
    assert data["role"] == "Project Lead & Backend Architect"


def test_team_member_missing_role_raises_error() -> None:
    """Test that TeamMember without role raises ValidationError.

    Arrange: Dict missing 'role'.
    Act & Assert: Validation raises ValidationError.
    """
    # Arrange
    incomplete = {"id": "P2"}

    # Act & Assert
    with pytest.raises(ValidationError):
        TeamMember.model_validate(incomplete)


# ── InfoResponse Tests ──────────────────────────────────────────────────


def test_info_response_serialization_to_dict() -> None:
    """Test serializing InfoResponse with nested models to dictionary.

    Arrange: InfoResponse instance with nested features and team members.
    Act: Dump model via model_dump().
    Assert: Dictionary includes all keys and correctly serialized nested lists.
    """
    # Arrange
    info = InfoResponse(
        version="0.1.0",
        environment="development",
        supported_source_languages=["en"],
        supported_target_languages=["te"],
        processing_mode="batch",
        features=[
            FeatureInfo(name="asr", status=FeatureStatus.NOT_IMPLEMENTED, description="ASR")
        ],
        team=[
            TeamMember(id="P1", role="Lead"),
        ],
    )

    # Act
    data = info.model_dump()

    # Assert
    assert data["version"] == "0.1.0"
    assert data["environment"] == "development"
    assert data["supported_source_languages"] == ["en"]
    assert data["supported_target_languages"] == ["te"]
    assert data["processing_mode"] == "batch"
    assert len(data["features"]) == 1
    assert data["features"][0]["name"] == "asr"
    assert len(data["team"]) == 1
    assert data["team"][0]["id"] == "P1"


def test_info_response_serialization_to_json() -> None:
    """Test serializing InfoResponse to JSON string.

    Arrange: InfoResponse instance.
    Act: Serialize to JSON via model_dump_json() and parse with json.loads.
    Assert: Resulting JSON document matches model structure.
    """
    # Arrange
    info = InfoResponse(
        version="0.1.0",
        environment="testing",
        supported_source_languages=["en"],
        supported_target_languages=["te"],
        processing_mode="batch",
        features=[],
        team=[],
    )

    # Act
    json_str = info.model_dump_json()
    parsed = json.loads(json_str)

    # Assert
    assert parsed["version"] == "0.1.0"
    assert parsed["environment"] == "testing"
    assert parsed["supported_source_languages"] == ["en"]
    assert parsed["supported_target_languages"] == ["te"]


def test_info_response_deserialization_from_dict() -> None:
    """Test deserializing raw dictionary into InfoResponse model.

    Arrange: Nested dictionary payload.
    Act: Validate via InfoResponse.model_validate.
    Assert: Validated model has correctly typed sub-models.
    """
    # Arrange
    raw = {
        "version": "0.1.0",
        "environment": "development",
        "supported_source_languages": ["en"],
        "supported_target_languages": ["te"],
        "processing_mode": "batch",
        "features": [
            {"name": "tts", "status": "not_implemented", "description": "TTS Engine"}
        ],
        "team": [
            {"id": "P3", "role": "TTS Engineer"}
        ],
    }

    # Act
    model = InfoResponse.model_validate(raw)

    # Assert
    assert model.version == "0.1.0"
    assert isinstance(model.features[0], FeatureInfo)
    assert model.features[0].status == FeatureStatus.NOT_IMPLEMENTED
    assert isinstance(model.team[0], TeamMember)
    assert model.team[0].id == "P3"


def test_info_response_missing_fields_raises_error() -> None:
    """Test that missing required fields in InfoResponse raises ValidationError.

    Arrange: Payload missing 'processing_mode' and 'features'.
    Act & Assert: Validation raises ValidationError.
    """
    # Arrange
    incomplete = {
        "version": "0.1.0",
        "environment": "development",
        "supported_source_languages": ["en"],
        "supported_target_languages": ["te"],
    }

    # Act & Assert
    with pytest.raises(ValidationError):
        InfoResponse.model_validate(incomplete)


# ── ErrorDetail and ErrorResponse Tests ─────────────────────────────────


def test_error_detail_serialization() -> None:
    """Test ErrorDetail serialization with field and message.

    Arrange: ErrorDetail instance with field and message.
    Act: Dump to dictionary.
    Assert: Matches provided field and message.
    """
    # Arrange
    detail = ErrorDetail(field="port", message="Input should be less than or equal to 65535")

    # Act
    data = detail.model_dump()

    # Assert
    assert data["field"] == "port"
    assert data["message"] == "Input should be less than or equal to 65535"


def test_error_detail_optional_field_defaults_to_none() -> None:
    """Test ErrorDetail allows field to default to None.

    Arrange: ErrorDetail without field name.
    Act: Inspect field attribute.
    Assert: field is None.
    """
    # Arrange & Act
    detail = ErrorDetail(message="Global request failure")

    # Assert
    assert detail.field is None
    assert detail.message == "Global request failure"


def test_error_response_serialization_with_defaults() -> None:
    """Test ErrorResponse serialization with default details and request_id.

    Arrange: ErrorResponse instance without explicit details or request_id.
    Act: Dump model to dict.
    Assert: details is empty list and request_id is '-'.
    """
    # Arrange
    error = ErrorResponse(
        error="not_found",
        message="Item not found",
        status_code=404,
    )

    # Act
    data = error.model_dump()

    # Assert
    assert data["error"] == "not_found"
    assert data["message"] == "Item not found"
    assert data["status_code"] == 404
    assert data["details"] == []
    assert data["request_id"] == "-"


def test_error_response_serialization_with_custom_details() -> None:
    """Test ErrorResponse serialization with custom ErrorDetail entries.

    Arrange: ErrorResponse with specific details and correlation request_id.
    Act: Dump model to dict and JSON.
    Assert: details list contains serialized ErrorDetail objects.
    """
    # Arrange
    error = ErrorResponse(
        error="validation_error",
        message="Request validation failed",
        status_code=422,
        details=[
            ErrorDetail(field="body -> audio", message="Field required"),
        ],
        request_id="req-abc-123",
    )

    # Act
    data = error.model_dump()
    json_data = json.loads(error.model_dump_json())

    # Assert
    assert data["status_code"] == 422
    assert data["request_id"] == "req-abc-123"
    assert len(data["details"]) == 1
    assert data["details"][0]["field"] == "body -> audio"
    assert json_data["details"][0]["message"] == "Field required"


def test_error_response_missing_required_fields_raises_error() -> None:
    """Test that missing required fields in ErrorResponse raises ValidationError.

    Arrange: Dict missing 'status_code'.
    Act & Assert: Validation raises ValidationError.
    """
    # Arrange
    incomplete = {
        "error": "server_error",
        "message": "Something went wrong",
    }

    # Act & Assert
    with pytest.raises(ValidationError):
        ErrorResponse.model_validate(incomplete)
