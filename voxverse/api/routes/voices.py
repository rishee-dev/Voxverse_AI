"""FastAPI router for speaker enrollment, profile retrieval, and deletion lifecycle.

Owner: P1 — Audio and Speaker Profile Engineer (reviewed with P4 — Pipeline Engineer)
"""

from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from voxverse.api.dependencies.auth import get_authenticated_user_id
from voxverse.api.dependencies.common import get_settings
from voxverse.config.logging import get_logger
from voxverse.config.settings import Settings
from voxverse.genai.speaker.exceptions import (
    ConsentRequiredError,
    ConsentRevokedError,
    EnrollmentValidationError,
    InvalidConsentError,
    PathTraversalError,
    ProfileDeletionError,
    ProfileNotFoundError,
    UnauthorizedProfileAccessError,
)
from voxverse.genai.speaker.schemas import (
    ConsentPurpose,
    DeletionResult,
    VoiceEnrollmentRequest,
    VoiceProfileMetadata,
)
from voxverse.genai.speaker.service import SpeakerProfileService
from voxverse.storage.voice_storage import VoiceProfileStorage

logger = get_logger("api.voices")

router = APIRouter(prefix="/api/v1/voices", tags=["Voices"])


def get_speaker_service(settings: Settings = Depends(get_settings)) -> SpeakerProfileService:
    """Dependency provider for SpeakerProfileService."""
    storage = VoiceProfileStorage(settings=settings)
    return SpeakerProfileService(storage=storage, settings=settings)


@router.post(
    "",
    response_model=VoiceProfileMetadata,
    status_code=status.HTTP_201_CREATED,
    summary="Enroll a new speaker voice profile with explicit affirmative consent.",
)
async def enroll_voice(
    file: UploadFile = File(..., description="Enrollment audio file (WAV, FLAC, MP3, OGG)."),
    display_name: str = Form(..., description="Human-readable voice profile label."),
    consent_confirmed: bool = Form(..., description="Explicit affirmation of consent for voice reference storage."),
    ownership_confirmed: bool = Form(..., description="Affirmation of recording ownership or explicit authorization."),
    purposes: str = Form(
        ...,
        description="Comma-separated or JSON list of authorized processing purposes (e.g., 'store_voice_reference,generate_speech').",
    ),
    consent_version: str = Form("1.0", description="Legal consent agreement version."),
    owner_id: str = Depends(get_authenticated_user_id),
    settings: Settings = Depends(get_settings),
    service: SpeakerProfileService = Depends(get_speaker_service),
) -> VoiceProfileMetadata:
    """Create a new speaker voice profile.

    Strict security invariants:
    - Requires authenticated caller identity (`owner_id`).
    - Requires explicit, affirmative consent (`consent_confirmed=true`).
    - Validates acoustic duration, RMS, silence, clipping, and human speech presence via Silero VAD.
    - Reference audio is stored securely and never exposed via public URL.
    """
    # 1. Parse purposes
    parsed_purposes: list[ConsentPurpose] = []
    try:
        clean_p = purposes.strip()
        if clean_p.startswith("[") and clean_p.endswith("]"):
            raw_list = json.loads(clean_p)
        else:
            raw_list = [p.strip() for p in clean_p.split(",") if p.strip()]

        for item in raw_list:
            parsed_purposes.append(ConsentPurpose(item))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid processing purposes declaration: {e}. Valid purposes: {[p.value for p in ConsentPurpose]}",
        )

    enrollment_request = VoiceEnrollmentRequest(
        owner_id=owner_id,
        display_name=display_name.strip(),
        consent_confirmed=consent_confirmed,
        ownership_confirmed=ownership_confirmed,
        purposes=parsed_purposes,
        consent_version=consent_version.strip(),
    )

    # 2. Save upload temporarily for validation
    temp_dir = settings.temp_path
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_upload = temp_dir / f"enroll_{uuid.uuid4().hex[:12]}_{file.filename or 'audio.wav'}"

    try:
        with open(temp_upload, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Check upload size limit
        file_size = temp_upload.stat().st_size
        if file_size > settings.max_audio_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Uploaded audio ({file_size} bytes) exceeds maximum limit of {settings.max_audio_size_mb} MB.",
            )

        # 3. Enroll voice profile
        try:
            profile_meta = service.enroll_voice(temp_upload, enrollment_request)
            return profile_meta
        except ConsentRequiredError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message) from e
        except InvalidConsentError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message) from e
        except EnrollmentValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"error": "enrollment_validation_failed", "issues": e.issues},
            ) from e
        except PathTraversalError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message) from e

    finally:
        # Clean up temporary uploaded file
        if temp_upload.exists():
            try:
                temp_upload.unlink()
            except OSError:
                pass


@router.get(
    "/{voice_id}",
    response_model=VoiceProfileMetadata,
    summary="Retrieve safe profile metadata for an enrolled voice.",
)
async def get_voice_profile(
    voice_id: str,
    owner_id: str = Depends(get_authenticated_user_id),
    service: SpeakerProfileService = Depends(get_speaker_service),
) -> VoiceProfileMetadata:
    """Retrieve safe metadata for a voice profile belonging to the authenticated owner."""
    try:
        return service.get_profile(voice_id, owner_id=owner_id)
    except ProfileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message) from e
    except UnauthorizedProfileAccessError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=e.message) from e
    except ConsentRevokedError as e:
        raise HTTPException(status_code=status.HTTP_410_GONE, detail=e.message) from e
    except PathTraversalError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message) from e


@router.get(
    "",
    response_model=list[VoiceProfileMetadata],
    summary="List all active voice profiles belonging to the authenticated owner.",
)
async def list_voice_profiles(
    owner_id: str = Depends(get_authenticated_user_id),
    service: SpeakerProfileService = Depends(get_speaker_service),
) -> list[VoiceProfileMetadata]:
    """List profiles owned by the caller."""
    try:
        return service.list_profiles(owner_id=owner_id)
    except PathTraversalError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message) from e


@router.delete(
    "/{voice_id}",
    response_model=DeletionResult,
    summary="Revoke consent and permanently delete reference audio and profile data.",
)
async def delete_voice_profile(
    voice_id: str,
    owner_id: str = Depends(get_authenticated_user_id),
    service: SpeakerProfileService = Depends(get_speaker_service),
) -> DeletionResult:
    """Revoke consent and delete all biometric audio files and metadata for this profile."""
    try:
        return service.revoke_consent_and_delete(voice_id, owner_id=owner_id)
    except ProfileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message) from e
    except UnauthorizedProfileAccessError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=e.message) from e
    except ProfileDeletionError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=e.message) from e
    except PathTraversalError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message) from e
