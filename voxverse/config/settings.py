"""Centralized configuration using Pydantic Settings.

All application settings are loaded from environment variables with the
``VOXVERSE_`` prefix.  A ``.env`` file in the project root is read
automatically when present.

Owner: Shared (all team members)
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Supported deployment environments."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Supported log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables.

    Every field maps to an environment variable with the ``VOXVERSE_`` prefix.
    For example ``VOXVERSE_APP_NAME`` → ``app_name``.

    Attributes are grouped by concern and documented individually.
    """

    model_config = SettingsConfigDict(
        env_prefix="VOXVERSE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────
    app_name: str = Field(
        default="VoxVerse",
        description="Human-readable application name.",
    )
    app_version: str = Field(
        default="0.1.0",
        description="Semantic version of the running application.",
    )
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Current deployment environment.",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode (verbose errors, auto-reload).",
    )

    # ── Server ───────────────────────────────────────────────────────────
    host: str = Field(
        default="0.0.0.0",
        description="Bind address for the HTTP server.",
    )
    port: Annotated[int, Field(ge=1, le=65535)] = Field(
        default=8000,
        description="Port the HTTP server listens on.",
    )
    workers: Annotated[int, Field(ge=1)] = Field(
        default=1,
        description="Number of Uvicorn worker processes.",
    )

    # ── CORS ─────────────────────────────────────────────────────────────
    cors_origins: list[str] | str = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins (list of URL strings or comma-separated string).",
    )
    cors_allow_credentials: bool = Field(
        default=True,
        description="Whether to allow credentials in CORS requests.",
    )
    cors_allow_methods: list[str] | str = Field(
        default=["*"],
        description="HTTP methods allowed in CORS requests.",
    )
    cors_allow_headers: list[str] | str = Field(
        default=["*"],
        description="HTTP headers allowed in CORS requests.",
    )

    # ── Logging ──────────────────────────────────────────────────────────
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Minimum log level.",
    )
    log_format: str = Field(
        default="text",
        description="Log output format: 'text' for human-readable, 'json' for structured.",
    )

    # ── File paths ───────────────────────────────────────────────────────
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2],
        description="Absolute path to the project root directory.",
    )
    upload_dir: str = Field(
        default="uploads",
        description="Directory name (relative to project_root) for uploaded files.",
    )
    output_dir: str = Field(
        default="outputs",
        description="Directory name (relative to project_root) for generated output files.",
    )
    temp_dir: str = Field(
        default="temp",
        description="Directory name (relative to project_root) for temporary files.",
    )
    storage_dir: str = Field(
        default="storage",
        description="Directory name (relative to project_root) for persistent storage.",
    )
    voices_dir: str = Field(
        default="voices",
        description="Subdirectory name inside storage_dir for speaker voice profiles.",
    )

    # ── Speaker Enrollment & Privacy ─────────────────────────────────────
    speaker_consent_required: bool = Field(
        default=True,
        description="Whether explicit affirmative consent is mandatory for speaker enrollment.",
    )
    min_enrollment_duration_seconds: float = Field(
        default=3.0,
        description="Minimum total duration in seconds for enrollment audio.",
    )
    max_enrollment_duration_seconds: float = Field(
        default=60.0,
        description="Maximum total duration in seconds for enrollment audio.",
    )
    min_speech_duration_seconds: float = Field(
        default=1.5,
        description="Minimum usable human speech duration in seconds detected via VAD.",
    )

    # ── Audio ────────────────────────────────────────────────────────────
    max_audio_size_mb: Annotated[int, Field(ge=1)] = Field(
        default=50,
        description="Maximum allowed audio file size in megabytes.",
    )
    allowed_audio_formats: list[str] = Field(
        default=["wav", "mp3", "flac", "ogg", "m4a", "webm"],
        description="Accepted audio file extensions (without dot).",
    )
    audio_sample_rate: Annotated[int, Field(ge=8000)] = Field(
        default=16000,
        description="Target sample rate (Hz) for audio preprocessing.",
    )

    # ── External services / API keys (empty by default) ──────────────────
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key (future use for Whisper API fallback).",
    )
    azure_speech_key: str = Field(
        default="",
        description="Azure Cognitive Services Speech key (future use).",
    )

    # ── Validators ───────────────────────────────────────────────────────

    @field_validator("log_format")
    @classmethod
    def _validate_log_format(cls, value: str) -> str:
        """Ensure log_format is one of the supported values."""
        allowed = {"text", "json"}
        if value.lower() not in allowed:
            msg = f"log_format must be one of {allowed}, got '{value}'"
            raise ValueError(msg)
        return value.lower()

    @field_validator("cors_origins", "cors_allow_methods", "cors_allow_headers", mode="before")
    @classmethod
    def _parse_list_or_str(cls, value: Any) -> list[str]:
        """Accept a comma-separated string, JSON list string, or a list."""
        if isinstance(value, str):
            val = value.strip()
            if val.startswith("[") and val.endswith("]"):
                try:
                    import json
                    parsed = json.loads(val)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed if str(item).strip()]
                except Exception:
                    pass
            return [origin.strip() for origin in val.split(",") if origin.strip()]
        if isinstance(value, (list, tuple)):
            return [str(origin).strip() for origin in value if str(origin).strip()]
        return value

    # ── Computed path helpers ────────────────────────────────────────────

    @property
    def upload_path(self) -> Path:
        """Absolute path to the upload directory."""
        return self.project_root / self.upload_dir

    @property
    def output_path(self) -> Path:
        """Absolute path to the output directory."""
        return self.project_root / self.output_dir

    @property
    def temp_path(self) -> Path:
        """Absolute path to the temporary directory."""
        return self.project_root / self.temp_dir

    @property
    def storage_path(self) -> Path:
        """Absolute path to the persistent storage directory."""
        return self.project_root / self.storage_dir

    @property
    def voices_path(self) -> Path:
        """Absolute path to the speaker voice profiles directory."""
        return self.storage_path / self.voices_dir

    @property
    def max_audio_size_bytes(self) -> int:
        """Maximum audio file size in bytes."""
        return self.max_audio_size_mb * 1024 * 1024

    def ensure_directories(self) -> None:
        """Create upload, output, temp, storage, and voices directories if they do not exist."""
        for path in (self.upload_path, self.output_path, self.temp_path, self.storage_path, self.voices_path):
            path.mkdir(parents=True, exist_ok=True)


# ── Module-level singleton accessor ─────────────────────────────────────

_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the cached application settings singleton.

    The instance is created on first call and reused afterwards.
    Override via environment variables or a ``.env`` file.
    """
    global _settings  # noqa: PLW0603
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Clear the cached settings singleton (useful in tests)."""
    global _settings  # noqa: PLW0603
    _settings = None
