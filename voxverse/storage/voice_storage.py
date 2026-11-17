"""Secure local storage abstraction for speaker voice profiles and reference audio.

Owner: P1 — Audio and Speaker Profile Engineer (reviewed with P4 — Pipeline Engineer)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import soundfile as sf

from voxverse.config.logging import get_logger
from voxverse.config.settings import Settings, get_settings
from voxverse.genai.speaker.exceptions import (
    PathTraversalError,
    ProfileDeletionError,
    ProfileNotFoundError,
    UnauthorizedProfileAccessError,
)
from voxverse.genai.speaker.schemas import (
    ConsentRecord,
    ConsentStatus,
    DeletionResult,
    EnrollmentStatus,
    VoiceProfile,
)

logger = get_logger("storage.voice")

SAFE_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class VoiceProfileStorage:
    """Manages persistent disk storage and deletion lifecycle for voice profiles.

    Security & Privacy Invariants:
    1. Strict Path Traversal Prevention: All IDs are validated with regex and path resolution.
    2. Atomic Writes: Metadata is written to temporary files and atomically renamed.
    3. Strict Biometric Segregation: Audio is stored as reference.wav, separate from metadata.
    4. Verified Irreversible Deletion: Biometric audio is deleted and verified absent from disk.
    5. No Path Leakage: Internal storage paths are never returned to public clients.
    """

    def __init__(self, base_dir: Path | str | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.base_dir = Path(base_dir).resolve() if base_dir else self.settings.voices_path.resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_id(self, identifier: str, param_name: str = "id") -> str:
        """Validate identifier against regex to prevent directory traversal."""
        clean = identifier.strip()
        if not clean or not SAFE_ID_REGEX.match(clean) or ".." in clean:
            logger.warning("Potential path traversal or invalid %s: '%s'", param_name, identifier)
            raise PathTraversalError(
                f"Invalid {param_name} format. Identifiers must be alphanumeric with no path separators.",
                code="INVALID_IDENTIFIER",
            )
        return clean

    def _get_profile_dir(self, profile_id: str) -> Path:
        """Resolve and verify profile directory remains strictly inside base storage."""
        clean_id = self._sanitize_id(profile_id, param_name="profile_id")
        target_dir = (self.base_dir / clean_id).resolve()
        try:
            target_dir.relative_to(self.base_dir)
        except ValueError as e:
            logger.error("Path traversal attempt detected: %s outside %s", target_dir, self.base_dir)
            raise PathTraversalError(
                "Access denied: Target directory escapes storage root.",
                code="PATH_TRAVERSAL_DETECTED",
            ) from e
        return target_dir

    def _atomic_write_json(self, destination: Path, data: dict[str, Any]) -> None:
        """Write JSON data via a temporary file and atomic replace."""
        temp_file = destination.with_suffix(".tmp")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            temp_file.replace(destination)
        except Exception as e:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            logger.error("Failed to write JSON atomically to %s: %s", destination, e)
            raise

    def save_profile(self, profile: VoiceProfile, audio_data: Any, sample_rate: int = 16000) -> Path:
        """Save a new voice profile, its consent declaration, and reference audio on disk.

        Returns
        -------
        Path
            Path to the written reference audio WAV file.
        """
        self._sanitize_id(profile.owner_id, param_name="owner_id")
        profile_dir = self._get_profile_dir(profile.profile_id)
        profile_dir.mkdir(parents=True, exist_ok=True)

        # 1. Write reference audio
        audio_dest = profile_dir / profile.reference_audio_filename
        sf.write(str(audio_dest), audio_data, sample_rate, subtype="PCM_16")

        # 2. Write consent record atomically
        consent_dest = profile_dir / "consent.json"
        self._atomic_write_json(consent_dest, profile.consent.model_dump(mode="json"))

        # 3. Write profile metadata atomically
        profile_dest = profile_dir / "profile.json"
        self._atomic_write_json(profile_dest, profile.model_dump(mode="json"))

        logger.info("Successfully stored voice profile %s for owner %s", profile.profile_id, profile.owner_id)
        return audio_dest

    def get_profile(self, profile_id: str, owner_id: str | None = None) -> VoiceProfile:
        """Retrieve a voice profile from disk with optional ownership verification."""
        clean_owner = self._sanitize_id(owner_id, param_name="owner_id") if owner_id is not None else None
        profile_dir = self._get_profile_dir(profile_id)
        profile_file = profile_dir / "profile.json"

        if not profile_file.exists():
            raise ProfileNotFoundError(
                f"Voice profile '{profile_id}' not found.",
                code="PROFILE_NOT_FOUND",
            )

        with open(profile_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        profile = VoiceProfile.model_validate(data)

        # Ownership verification
        if clean_owner is not None and profile.owner_id != clean_owner:
            logger.warning(
                "Unauthorized access attempt to profile %s by user %s (owner: %s)",
                profile_id,
                clean_owner,
                profile.owner_id,
            )
            raise UnauthorizedProfileAccessError(
                "You do not have authorization to view this voice profile.",
                code="UNAUTHORIZED_PROFILE_ACCESS",
            )

        return profile

    def list_profiles(self, owner_id: str) -> list[VoiceProfile]:
        """List all active voice profiles belonging to the authenticated owner."""
        clean_owner = self._sanitize_id(owner_id, param_name="owner_id")
        results: list[VoiceProfile] = []

        if not self.base_dir.exists():
            return results

        for child in self.base_dir.iterdir():
            if child.is_dir():
                profile_file = child / "profile.json"
                if profile_file.exists():
                    try:
                        with open(profile_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        prof = VoiceProfile.model_validate(data)
                        if prof.owner_id == clean_owner and prof.enrollment_status != EnrollmentStatus.DELETED:
                            results.append(prof)
                    except Exception as e:
                        logger.warning("Skipping invalid profile in %s: %s", child.name, e)

        return results

    def get_reference_audio_path(self, profile_id: str, owner_id: str | None = None) -> Path:
        """Return the internal path to reference audio (for internal ML pipeline use only)."""
        profile = self.get_profile(profile_id, owner_id=owner_id)
        profile_dir = self._get_profile_dir(profile_id)
        audio_path = profile_dir / profile.reference_audio_filename

        if not audio_path.exists():
            raise ProfileNotFoundError(
                f"Reference audio for profile '{profile_id}' is missing from storage.",
                code="REFERENCE_AUDIO_MISSING",
            )
        return audio_path

    def delete_profile(self, profile_id: str, owner_id: str) -> DeletionResult:
        """Irreversibly delete reference audio and profile metadata, verifying disk removal."""
        clean_owner = self._sanitize_id(owner_id, param_name="owner_id")
        profile = self.get_profile(profile_id, owner_id=clean_owner)
        profile_dir = self._get_profile_dir(profile_id)

        now = datetime.now(timezone.utc)
        audio_file = profile_dir / profile.reference_audio_filename
        profile_file = profile_dir / "profile.json"
        consent_file = profile_dir / "consent.json"

        # 1. Delete reference audio file
        ref_deleted = False
        if audio_file.exists():
            audio_file.unlink()
            ref_deleted = not audio_file.exists()
        else:
            ref_deleted = True

        # 2. Delete any derived artifacts or cache in directory
        for item in profile_dir.iterdir():
            if item.name not in {"profile.json", "consent.json", "tombstone.json"}:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)

        # 3. Write minimal tombstone record for audit trail (contains ZERO biometric or audio data)
        tombstone_data = {
            "profile_id": profile_id,
            "owner_id": clean_owner,
            "deleted_at": now.isoformat(),
            "consent_status": ConsentStatus.REVOKED.value,
            "biometric_audio_deleted": ref_deleted,
        }
        self._atomic_write_json(profile_dir / "tombstone.json", tombstone_data)

        # 4. Remove active profile.json and consent.json
        meta_deleted = False
        if profile_file.exists():
            profile_file.unlink()
            meta_deleted = not profile_file.exists()
        else:
            meta_deleted = True

        if consent_file.exists():
            consent_file.unlink()

        # 5. Verify deletion
        all_verified = ref_deleted and meta_deleted and (not audio_file.exists()) and (not profile_file.exists())

        if not all_verified:
            logger.critical("Profile deletion failed to verify absence on disk for %s", profile_id)
            raise ProfileDeletionError(
                f"Deletion verification failed on disk for profile '{profile_id}'. "
                "Some biometric or metadata files could not be confirmed removed.",
                code="DELETION_VERIFICATION_FAILED",
            )

        logger.info("Irreversible deletion verified for profile %s (owner: %s)", profile_id, clean_owner)
        return DeletionResult(
            profile_id=profile_id,
            owner_id=clean_owner,
            deleted_at=now,
            reference_audio_deleted=ref_deleted,
            normalized_audio_deleted=ref_deleted,
            metadata_deleted=meta_deleted,
            consent_revoked=True,
            all_verified_deleted=True,
            details="All reference audio, derived artifacts, and active metadata verified deleted.",
        )
