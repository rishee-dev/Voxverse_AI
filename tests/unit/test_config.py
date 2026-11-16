"""Unit tests for VoxVerse application configuration (Settings).

Covers:
- Default configuration values
- Loading and overriding configuration from environment variables (VOXVERSE_ prefix)
- Port validation (1-65535 range boundaries)
- Worker process validation (ge=1)
- Audio sample rate and file size constraints
- Log format validation ('text' or 'json' only)
- CORS origin parsing (comma-separated string to list)
- Computed path properties and directory creation helper
- Singleton lifecycle (get_settings, reset_settings)

Owner: Shared (all team members)
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from voxverse.config.settings import (
    Environment,
    LogLevel,
    Settings,
    get_settings,
    reset_settings,
)


def test_default_settings_values() -> None:
    """Test that default settings match Phase 0 specifications.

    Arrange: Clear settings and instantiate default Settings.
    Act: Inspect default attributes.
    Assert: All defaults match documented MVP specifications.
    """
    # Arrange & Act
    settings = Settings()

    # Assert
    assert settings.app_name == "VoxVerse"
    assert settings.app_version == "0.1.0"
    assert settings.environment == Environment.DEVELOPMENT
    assert settings.debug is False
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.workers == 1
    assert settings.cors_origins == ["http://localhost:3000", "http://localhost:8000"]
    assert settings.cors_allow_credentials is True
    assert settings.cors_allow_methods == ["*"]
    assert settings.cors_allow_headers == ["*"]
    assert settings.log_level == LogLevel.INFO
    assert settings.log_format == "text"
    assert settings.upload_dir == "uploads"
    assert settings.output_dir == "outputs"
    assert settings.temp_dir == "temp"
    assert settings.max_audio_size_mb == 50
    assert settings.audio_sample_rate == 16000
    assert "wav" in settings.allowed_audio_formats
    assert "mp3" in settings.allowed_audio_formats
    assert settings.openai_api_key == ""
    assert settings.azure_speech_key == ""


def test_settings_load_from_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that settings correctly load values from VOXVERSE_ prefixed environment variables.

    Arrange: Set custom environment variables using monkeypatch.
    Act: Instantiate a new Settings object.
    Assert: Settings attributes reflect the environment variable values.
    """
    # Arrange
    monkeypatch.setenv("VOXVERSE_APP_NAME", "CustomVox")
    monkeypatch.setenv("VOXVERSE_APP_VERSION", "1.2.3")
    monkeypatch.setenv("VOXVERSE_PORT", "9090")
    monkeypatch.setenv("VOXVERSE_ENVIRONMENT", "production")
    monkeypatch.setenv("VOXVERSE_DEBUG", "true")
    monkeypatch.setenv("VOXVERSE_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("VOXVERSE_LOG_FORMAT", "json")
    monkeypatch.setenv("VOXVERSE_MAX_AUDIO_SIZE_MB", "100")
    monkeypatch.setenv("VOXVERSE_AUDIO_SAMPLE_RATE", "22050")

    # Act
    settings = Settings()

    # Assert
    assert settings.app_name == "CustomVox"
    assert settings.app_version == "1.2.3"
    assert settings.port == 9090
    assert settings.environment == Environment.PRODUCTION
    assert settings.debug is True
    assert settings.log_level == LogLevel.DEBUG
    assert settings.log_format == "json"
    assert settings.max_audio_size_mb == 100
    assert settings.audio_sample_rate == 22050


def test_settings_port_lower_bound_validation() -> None:
    """Test that port numbers below 1 trigger a ValidationError.

    Arrange: Target port=0 (invalid).
    Act & Assert: Instantiating Settings with port=0 raises ValidationError.
    """
    # Arrange
    invalid_port = 0

    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        Settings(port=invalid_port)
    errors = exc_info.value.errors()
    assert any("port" in str(err["loc"]) for err in errors)


def test_settings_port_upper_bound_validation() -> None:
    """Test that port numbers above 65535 trigger a ValidationError.

    Arrange: Target port=70000 (invalid).
    Act & Assert: Instantiating Settings with port=70000 raises ValidationError.
    """
    # Arrange
    invalid_port = 70000

    # Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        Settings(port=invalid_port)
    errors = exc_info.value.errors()
    assert any("port" in str(err["loc"]) for err in errors)


def test_settings_valid_port_range_edges() -> None:
    """Test that valid boundary port numbers (1 and 65535) are accepted.

    Arrange: Edge port values 1 and 65535.
    Act: Instantiate Settings with each boundary value.
    Assert: Ports are accepted without error.
    """
    # Arrange & Act
    settings_min = Settings(port=1)
    settings_max = Settings(port=65535)

    # Assert
    assert settings_min.port == 1
    assert settings_max.port == 65535


def test_settings_workers_validation() -> None:
    """Test that workers count less than 1 raises ValidationError.

    Arrange: Invalid workers=0.
    Act & Assert: ValidationError is raised.
    """
    # Arrange & Act & Assert
    with pytest.raises(ValidationError):
        Settings(workers=0)


def test_settings_max_audio_size_validation() -> None:
    """Test that max_audio_size_mb less than 1 raises ValidationError.

    Arrange: Invalid max_audio_size_mb=0.
    Act & Assert: ValidationError is raised.
    """
    # Arrange & Act & Assert
    with pytest.raises(ValidationError):
        Settings(max_audio_size_mb=0)


def test_settings_audio_sample_rate_validation() -> None:
    """Test that audio_sample_rate below 8000 raises ValidationError.

    Arrange: Invalid audio_sample_rate=4000.
    Act & Assert: ValidationError is raised.
    """
    # Arrange & Act & Assert
    with pytest.raises(ValidationError):
        Settings(audio_sample_rate=4000)


def test_settings_log_format_validation_valid() -> None:
    """Test that valid log formats ('text', 'json', uppercase) are accepted.

    Arrange: Various valid case variations of log format strings.
    Act: Instantiate Settings with 'TEXT' and 'JSON'.
    Assert: Lowercase string is normalized and stored.
    """
    # Arrange & Act
    settings_text = Settings(log_format="TEXT")
    settings_json = Settings(log_format="JSON")

    # Assert
    assert settings_text.log_format == "text"
    assert settings_json.log_format == "json"


def test_settings_log_format_validation_invalid() -> None:
    """Test that unsupported log formats (e.g. 'yaml') trigger ValidationError.

    Arrange: Unsupported log format 'yaml'.
    Act & Assert: Instantiating Settings raises ValidationError.
    """
    # Arrange & Act & Assert
    with pytest.raises(ValidationError) as exc_info:
        Settings(log_format="yaml")
    assert "log_format" in str(exc_info.value)


def test_settings_cors_origins_from_comma_separated_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that cors_origins accepts a comma-separated string and parses to a list.

    Arrange: Supply comma-separated origins string via environment.
    Act: Instantiate Settings.
    Assert: cors_origins is parsed into a trimmed list of strings.
    """
    # Arrange
    raw_origins = "http://frontend.local:3000, https://voxverse.example.com "
    monkeypatch.setenv("VOXVERSE_CORS_ORIGINS", raw_origins)

    # Act
    settings = Settings()

    # Assert
    assert settings.cors_origins == [
        "http://frontend.local:3000",
        "https://voxverse.example.com",
    ]


def test_settings_cors_origins_from_list() -> None:
    """Test that cors_origins accepts an explicit list of origins directly.

    Arrange: List of origin URLs.
    Act: Instantiate Settings with list.
    Assert: cors_origins retains the provided list.
    """
    # Arrange
    origins_list = ["https://app1.internal", "https://app2.internal"]

    # Act
    settings = Settings(cors_origins=origins_list)

    # Assert
    assert settings.cors_origins == origins_list


def test_settings_computed_paths() -> None:
    """Test computed path properties (upload_path, output_path, temp_path).

    Arrange: Settings instance with known directories.
    Act: Access path properties.
    Assert: Paths resolve under project_root with expected subdirectories.
    """
    # Arrange
    settings = Settings(
        upload_dir="custom_uploads",
        output_dir="custom_outputs",
        temp_dir="custom_temp",
    )

    # Act & Assert
    assert settings.upload_path == settings.project_root / "custom_uploads"
    assert settings.output_path == settings.project_root / "custom_outputs"
    assert settings.temp_path == settings.project_root / "custom_temp"


def test_settings_max_audio_size_bytes_calculation() -> None:
    """Test that max_audio_size_bytes accurately converts MB to bytes.

    Arrange: max_audio_size_mb set to 10 MB.
    Act: Read max_audio_size_bytes property.
    Assert: Equals 10 * 1024 * 1024 bytes.
    """
    # Arrange
    settings = Settings(max_audio_size_mb=10)

    # Act
    size_bytes = settings.max_audio_size_bytes

    # Assert
    assert size_bytes == 10 * 1024 * 1024


def test_settings_ensure_directories(tmp_path: Path) -> None:
    """Test that ensure_directories creates upload, output, and temp directories.

    Arrange: Settings configured with project_root pointing to isolated tmp_path.
    Act: Invoke ensure_directories().
    Assert: upload_path, output_path, and temp_path directories exist on disk.
    """
    # Arrange
    settings = Settings(
        project_root=tmp_path,
        upload_dir="up",
        output_dir="out",
        temp_dir="tmp",
    )

    # Act
    settings.ensure_directories()

    # Assert
    assert (tmp_path / "up").is_dir()
    assert (tmp_path / "out").is_dir()
    assert (tmp_path / "tmp").is_dir()


def test_get_settings_singleton_caching() -> None:
    """Test that get_settings returns the same cached instance across invocations.

    Arrange: Reset singleton.
    Act: Call get_settings() twice.
    Assert: Both return references to the identical object.
    """
    # Arrange
    reset_settings()

    # Act
    instance_one = get_settings()
    instance_two = get_settings()

    # Assert
    assert instance_one is instance_two


def test_reset_settings_clears_singleton() -> None:
    """Test that reset_settings clears the cached singleton.

    Arrange: Obtain an initial settings singleton.
    Act: Call reset_settings() then get_settings() again.
    Assert: A new, distinct instance is returned.
    """
    # Arrange
    instance_old = get_settings()

    # Act
    reset_settings()
    instance_new = get_settings()

    # Assert
    assert instance_old is not instance_new
