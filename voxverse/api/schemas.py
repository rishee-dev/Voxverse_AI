"""Shared Pydantic schemas for VoxVerse API request/response models.

All API routes should use these models (or subclasses) for consistent
serialisation and OpenAPI documentation.

Owner: Shared (all team members)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enumerations ────────────────────────────────────────────────────────


class FeatureStatus(str, Enum):
    """Readiness level of a pipeline feature."""

    NOT_IMPLEMENTED = "not_implemented"
    EXPERIMENTAL = "experimental"
    STABLE = "stable"


# ── Response models ─────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Response body for the ``GET /health`` endpoint."""

    status: str = Field(
        ...,
        description="Current health status of the service.",
        examples=["healthy"],
    )
    version: str = Field(
        ...,
        description="Semantic version of the running application.",
        examples=["0.1.0"],
    )
    timestamp: datetime = Field(
        ...,
        description="Server UTC timestamp at the time of the response.",
    )


class FeatureInfo(BaseModel):
    """Describes one pipeline feature and its readiness."""

    name: str = Field(..., description="Machine-readable feature name.")
    status: FeatureStatus = Field(
        ...,
        description="Current implementation status.",
    )
    description: str = Field(
        default="",
        description="Short human-readable summary of the feature.",
    )


class TeamMember(BaseModel):
    """Describes one team member's role in the project."""

    id: str = Field(..., description="Short identifier, e.g. 'P1'.")
    role: str = Field(..., description="Team member's responsibility.")


class InfoResponse(BaseModel):
    """Response body for the ``GET /api/v1/info`` endpoint."""

    version: str = Field(..., description="Application version.")
    environment: str = Field(
        ...,
        description="Deployment environment name.",
        examples=["development"],
    )
    supported_source_languages: list[str] = Field(
        ...,
        description="ISO 639-1 codes for supported source languages.",
    )
    supported_target_languages: list[str] = Field(
        ...,
        description="ISO 639-1 codes for supported target languages.",
    )
    processing_mode: str = Field(
        ...,
        description="Audio processing strategy ('batch' or 'streaming').",
        examples=["batch"],
    )
    features: list[FeatureInfo] = Field(
        ...,
        description="Pipeline features and their implementation status.",
    )
    team: list[TeamMember] = Field(
        ...,
        description="Project team members and their roles.",
    )


class ErrorDetail(BaseModel):
    """A single error detail entry (loosely modelled on RFC 7807)."""

    field: str | None = Field(
        default=None,
        description="Name of the field that caused the error, if applicable.",
    )
    message: str = Field(..., description="Human-readable error message.")


class ErrorResponse(BaseModel):
    """Standard error response body returned by all error handlers.

    Follows a simplified Problem Details (RFC 7807) structure.
    """

    error: str = Field(
        ...,
        description="Short error code or title.",
        examples=["not_found"],
    )
    message: str = Field(
        ...,
        description="Human-readable description of the error.",
    )
    status_code: int = Field(
        ...,
        description="HTTP status code.",
        examples=[404],
    )
    details: list[ErrorDetail] = Field(
        default_factory=list,
        description="Optional list of granular error details.",
    )
    request_id: str = Field(
        default="-",
        description="Correlation ID for the request that triggered the error.",
    )
