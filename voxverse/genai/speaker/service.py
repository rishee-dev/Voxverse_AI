"""Speaker profile service coordinating consent, enrollment validation, and storage.

Owner: P1 — Audio and Speaker Profile Engineer
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from voxverse.config.logging import get_logger
from voxverse.config.settings import Settings, get_settings
from voxverse.genai.audio.normalizer import AudioNormalizer
from voxverse.genai.speaker.exceptions import (
    ConsentRequiredError,
    ConsentRevokedError,
    InvalidConsentError,
)
from voxverse.genai.speaker.schemas import (
    ConsentPurpose,
    ConsentRecord,
    ConsentStatus,
    DeletionResult,
    EnrollmentStatus,
    VoiceEnrollmentRequest,
    VoiceProfile,
    VoiceProfileMetadata,
)
from voxverse.genai.speaker.validator import EnrollmentValidator
from voxverse.storage.voice_storage import VoiceProfileStorage

logger = get_logger("speaker.service")


class SpeakerProfileService:
    """Orchestrates speaker enrollment, affirmative consent, profile retrieval, and deletion.

    Invariants:
    1. Zero Unconsented Enrollment: Every profile requires explicit, affirmative consent.
    2. Purpose Separation: Audio storage consent is distinct from speech generation consent.
    3. Biometric Verification: Enrollment audio must contain verified human speech.
    4. Authenticated Ownership: Every profile operation strictly verifies owner_id.
    5. Clean Deletion: Revoking consent triggers immediate deletion and disk absence verification.
    """

    def __init__(
        self,
        storage: VoiceProfileStorage | None = None,
        validator: EnrollmentValidator | None = None,
        normalizer: AudioNormalizer | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.storage = storage or VoiceProfileStorage(settings=self.settings)
        self.validator = validator or EnrollmentValidator(settings=self.settings)
        self.normalizer = normalizer or AudioNormalizer()

    def _verify_consent_request(self, request: VoiceEnrollmentRequest) -> None:
        """Verify that an enrollment request satisfies all legal and operational consent invariants."""
        if not self.settings.speaker_consent_required:
            return

        if not request.consent_confirmed:
            logger.warning("Enrollment rejected: consent_confirmed is False for owner %s", request.owner_id)
            raise ConsentRequiredError(
                "Speaker enrollment requires explicit affirmative consent (consent_confirmed=true).",
                code="CONSENT_CONFIRMATION_REQUIRED",
            )

        if not request.ownership_confirmed:
            logger.warning("Enrollment rejected: ownership_confirmed is False for owner %s", request.owner_id)
            raise InvalidConsentError(
                "You must affirm that you own the reference recording or have explicit authorization (ownership_confirmed=true).",
                code="OWNERSHIP_CONFIRMATION_REQUIRED",
            )

        if not request.purposes:
            logger.warning("Enrollment rejected: no purposes authorized for owner %s", request.owner_id)
            raise InvalidConsentError(
                "At least one specific processing purpose must be granted.",
                code="PURPOSES_REQUIRED",
            )

        if ConsentPurpose.STORE_VOICE_REFERENCE not in request.purposes:
            logger.warning("Enrollment rejected: STORE_VOICE_REFERENCE omitted in purposes for owner %s", request.owner_id)
            raise InvalidConsentError(
                "Processing purpose 'store_voice_reference' must be explicitly granted for enrollment.",
                code="STORE_PURPOSE_REQUIRED",
            )

    def enroll_voice(
        self,
        audio_path: Path | str,
        request: VoiceEnrollmentRequest,
    ) -> VoiceProfileMetadata:
        """Enroll a new speaker voice profile with validated consent and acoustic checks.

        Parameters
        ----------
        audio_path:
            Path to the enrollment recording file.
        request:
            Payload specifying owner ID, display name, and affirmative consent declarations.

        Returns
        -------
        VoiceProfileMetadata
            Safe public metadata for the newly created voice profile.
        """
        source_path = Path(audio_path).resolve()
        logger.info("Enrolling voice profile for owner '%s' from %s", request.owner_id, source_path.name)

        # 1. Enforce affirmative consent invariants
        self._verify_consent_request(request)

        # 2. Validate acoustic quality and human speech presence via P1 validator
        decoded_audio, original_sr, validation_report = self.validator.validate_enrollment_audio(source_path)

        # 3. Standardize audio to canonical 16 kHz mono 16-bit PCM WAV
        mono_audio = np.mean(decoded_audio, axis=1, dtype=np.float32) if decoded_audio.ndim > 1 else decoded_audio.astype(np.float32)

        if original_sr != 16000:
            import scipy.signal as signal
            gcd = np.gcd(16000, original_sr)
            up = 16000 // gcd
            down = original_sr // gcd
            normalized_16k = signal.resample_poly(mono_audio, up, down).astype(np.float32)
        else:
            normalized_16k = mono_audio

        # 4. Generate cryptographically safe IDs
        profile_id = f"spk_{uuid.uuid4().hex[:16]}"
        now = datetime.now(timezone.utc)

        # 5. Create immutable consent record
        consent = ConsentRecord(
            speaker_id=profile_id,
            owner_id=request.owner_id,
            status=ConsentStatus.GRANTED,
            granted_at=now,
            consent_version=request.consent_version,
            purposes=request.purposes,
            recording_ownership_confirmed=request.ownership_confirmed,
            declared_purpose=request.declared_purpose,
        )

        # 6. Assemble complete internal profile
        profile = VoiceProfile(
            profile_id=profile_id,
            owner_id=request.owner_id,
            display_name=request.display_name,
            created_at=now,
            updated_at=now,
            consent=consent,
            validation_report=validation_report,
            reference_audio_filename="reference.wav",
            audio_format="WAV",
            enrollment_status=EnrollmentStatus.ENROLLED,
        )

        # 7. Persist to storage
        self.storage.save_profile(profile, normalized_16k, sample_rate=16000)

        logger.info(
            "Voice profile %s enrolled successfully (duration=%.2fs, speech=%.2fs)",
            profile_id,
            validation_report.duration_seconds,
            validation_report.detected_speech_duration_seconds,
        )
        return profile.to_metadata()

    def get_profile(self, profile_id: str, owner_id: str) -> VoiceProfileMetadata:
        """Retrieve safe public metadata for an enrolled profile after verifying ownership."""
        profile = self.storage.get_profile(profile_id, owner_id=owner_id)
        if profile.consent.status == ConsentStatus.REVOKED:
            raise ConsentRevokedError(
                f"Consent for profile '{profile_id}' has been revoked. Profile is disabled.",
                code="CONSENT_REVOKED",
            )
        return profile.to_metadata()

    def list_profiles(self, owner_id: str) -> list[VoiceProfileMetadata]:
        """List all active profiles belonging to the authenticated owner."""
        profiles = self.storage.list_profiles(owner_id=owner_id)
        return [p.to_metadata() for p in profiles if p.consent.status != ConsentStatus.REVOKED]

    def revoke_consent_and_delete(self, profile_id: str, owner_id: str) -> DeletionResult:
        """Revoke speaker consent and permanently delete all associated reference audio and metadata."""
        logger.info("Revoking consent and deleting profile %s for owner %s", profile_id, owner_id)
        return self.storage.delete_profile(profile_id, owner_id=owner_id)

    def get_reference_audio_path(self, profile_id: str, owner_id: str) -> Path:
        """Internal pipeline helper to retrieve path to reference audio."""
        profile = self.storage.get_profile(profile_id, owner_id=owner_id)
        if profile.consent.status == ConsentStatus.REVOKED:
            raise ConsentRevokedError(
                f"Consent for profile '{profile_id}' was revoked. Reference audio cannot be accessed.",
                code="CONSENT_REVOKED",
            )
        return self.storage.get_reference_audio_path(profile_id, owner_id=owner_id)
