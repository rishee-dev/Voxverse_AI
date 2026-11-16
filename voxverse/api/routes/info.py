"""System information endpoint.

Exposes build metadata, supported languages, pipeline feature statuses,
and team ownership information.

Owner: Shared (all team members)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from voxverse.api.dependencies.common import get_settings
from voxverse.api.schemas import (
    FeatureInfo,
    FeatureStatus,
    InfoResponse,
    TeamMember,
)
from voxverse.config.settings import Settings

router = APIRouter(prefix="/api/v1", tags=["info"])

# ── Static data ─────────────────────────────────────────────────────────

_FEATURES: list[FeatureInfo] = [
    FeatureInfo(
        name="audio_upload",
        status=FeatureStatus.NOT_IMPLEMENTED,
        description="Upload English audio files for processing.",
    ),
    FeatureInfo(
        name="transcription",
        status=FeatureStatus.NOT_IMPLEMENTED,
        description="Transcribe English audio to text using Whisper.",
    ),
    FeatureInfo(
        name="translation",
        status=FeatureStatus.NOT_IMPLEMENTED,
        description="Translate English text to Telugu.",
    ),
    FeatureInfo(
        name="voice_cloning",
        status=FeatureStatus.NOT_IMPLEMENTED,
        description="Clone speaker voice characteristics for TTS.",
    ),
    FeatureInfo(
        name="tts",
        status=FeatureStatus.NOT_IMPLEMENTED,
        description="Synthesise Telugu speech from translated text.",
    ),
]

_TEAM: list[TeamMember] = [
    TeamMember(id="P1", role="Audio and Speaker Profile Engineer"),
    TeamMember(id="P2", role="Speech Understanding Engineer"),
    TeamMember(id="P3", role="Voice Generation Engineer"),
    TeamMember(id="P4", role="Pipeline Engineer"),
]


# ── Route ───────────────────────────────────────────────────────────────


@router.get(
    "/info",
    response_model=InfoResponse,
    summary="System information",
    description=(
        "Returns application metadata including version, supported languages, "
        "feature implementation status, and team roles."
    ),
)
async def system_info(
    settings: Settings = Depends(get_settings),
) -> InfoResponse:
    """Build and return the system information payload."""
    return InfoResponse(
        version=settings.app_version,
        environment=settings.environment.value,
        supported_source_languages=["en"],
        supported_target_languages=["te"],
        processing_mode="batch",
        features=_FEATURES,
        team=_TEAM,
    )
