"""Data models and schemas for speaker enrollment, consent, and voice profiles.

Owner: P1 — Audio and Speaker Profile Engineer
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field


class ConsentPurpose(str, Enum):
    """Specific, granular authorizations for voice data usage."""

    STORE_VOICE_REFERENCE = "store_voice_reference"
    GENERATE_SPEECH = "generate_speech"


class ConsentStatus(str, Enum):
    """Status of affirmative speaker consent."""

    GRANTED = "granted"
    REVOKED = "revoked"


class ConsentRecord(BaseModel):
    """Cryptographically verifiable affirmative consent record."""

    consent_id: str = Field(
        default_factory=lambda: f"cst_{uuid.uuid4().hex[:16]}",
        description="Unique identifier for the consent declaration.",
    )
    speaker_id: str = Field(..., description="Target speaker profile ID.")
    owner_id: str = Field(..., description="Authenticated user account ID owning the profile.")
    status: ConsentStatus = Field(
        default=ConsentStatus.GRANTED,
        description="Current state of consent.",
    )
    granted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp in UTC when affirmative consent was recorded.",
    )
    revoked_at: datetime | None = Field(
        default=None,
        description="Timestamp in UTC when consent was revoked, if applicable.",
    )
    consent_version: str = Field(
        default="1.0",
        description="Version of legal consent terms agreed to.",
    )
    purposes: list[ConsentPurpose] = Field(
        ...,
        description="List of specific granted processing purposes.",
    )
    recording_ownership_confirmed: bool = Field(
        ...,
        description="Affirmative declaration that the uploader is the speaker or has explicit rights.",
    )
    declared_purpose: str = Field(
        default="Voice reference storage and speaker-conditioned speech translation",
        description="Plain-text declaration of processing scope.",
    )


class EnrollmentStatus(str, Enum):
    """Lifecycle state of a voice profile."""

    PENDING = "pending"
    ENROLLED = "enrolled"
    REVOKED = "revoked"
    DELETED = "deleted"


class EnrollmentValidationReport(BaseModel):
    """Quality and acoustic validation assessment for enrollment audio."""

    is_valid: bool = Field(..., description="True if audio meets all enrollment criteria.")
    duration_seconds: float = Field(..., description="Total duration of enrollment audio.")
    sample_rate: int = Field(..., description="Audio sample rate in Hertz.")
    channels: int = Field(..., description="Audio channel count.")
    rms_amplitude: float = Field(..., description="Measured Root-Mean-Square amplitude.")
    peak_amplitude: float = Field(..., description="Measured absolute peak amplitude.")
    is_silent: bool = Field(..., description="True if audio is classified as silence.")
    is_clipped: bool = Field(..., description="True if audio exhibits digital clipping.")
    detected_speech_duration_seconds: float = Field(
        ...,
        description="Measured speech duration detected via Silero VAD.",
    )
    speech_ratio: float = Field(
        ...,
        description="Ratio of speech duration to total duration.",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="List of detected anomalies or rejection reasons.",
    )


class VoiceProfileMetadata(BaseModel):
    """Safe, non-sensitive public metadata for a voice profile.

    Strictly excludes internal filesystem paths and raw biometric/audio bytes.
    """

    profile_id: str = Field(..., description="Unique voice profile identifier.")
    owner_id: str = Field(..., description="Owner account identifier.")
    display_name: str = Field(..., description="Human-readable voice label.")
    created_at: datetime = Field(..., description="Enrollment creation timestamp in UTC.")
    updated_at: datetime = Field(..., description="Last modification timestamp in UTC.")
    consent_status: ConsentStatus = Field(..., description="Current consent status.")
    consent_version: str = Field(..., description="Agreed consent version.")
    consented_purposes: list[ConsentPurpose] = Field(..., description="Authorized purposes.")
    audio_duration_seconds: float = Field(..., description="Duration of reference recording.")
    usable_speech_duration_seconds: float = Field(..., description="Acoustically validated speech duration.")
    enrollment_status: EnrollmentStatus = Field(..., description="Current profile lifecycle state.")


class VoiceProfile(BaseModel):
    """Internal complete representation of an enrolled voice profile on disk."""

    profile_id: str
    owner_id: str
    display_name: str
    created_at: datetime
    updated_at: datetime
    consent: ConsentRecord
    validation_report: EnrollmentValidationReport
    reference_audio_filename: str = "reference.wav"
    audio_format: str = "WAV"
    enrollment_status: EnrollmentStatus = EnrollmentStatus.ENROLLED

    def to_metadata(self) -> VoiceProfileMetadata:
        """Export safe public metadata object with no internal storage paths."""
        return VoiceProfileMetadata(
            profile_id=self.profile_id,
            owner_id=self.owner_id,
            display_name=self.display_name,
            created_at=self.created_at,
            updated_at=self.updated_at,
            consent_status=self.consent.status,
            consent_version=self.consent.consent_version,
            consented_purposes=self.consent.purposes,
            audio_duration_seconds=self.validation_report.duration_seconds,
            usable_speech_duration_seconds=self.validation_report.detected_speech_duration_seconds,
            enrollment_status=self.enrollment_status,
        )


class VoiceEnrollmentRequest(BaseModel):
    """Payload declaring consent and metadata for new voice profile enrollment."""

    owner_id: str = Field(..., description="Authenticated user account ID.")
    display_name: str = Field(..., description="Label for the enrolled speaker voice.")
    consent_confirmed: bool = Field(
        ...,
        description="Affirmative confirmation of consent for voice reference storage.",
    )
    ownership_confirmed: bool = Field(
        ...,
        description="Confirmation that uploader owns the recording or has explicit authorization.",
    )
    purposes: list[ConsentPurpose] = Field(
        ...,
        description="List of requested processing authorizations.",
    )
    consent_version: str = Field(
        default="1.0",
        description="Terms version agreed to.",
    )
    declared_purpose: str = Field(
        default="Voice reference storage and speaker-conditioned speech translation",
        description="Declared purpose for audit trail.",
    )


class DeletionResult(BaseModel):
    """Verification outcome of profile and biometric data deletion."""

    profile_id: str = Field(..., description="Target profile ID.")
    owner_id: str = Field(..., description="Profile owner ID.")
    deleted_at: datetime = Field(..., description="Deletion timestamp in UTC.")
    reference_audio_deleted: bool = Field(..., description="True if reference.wav is absent.")
    metadata_deleted: bool = Field(..., description="True if profile.json is absent.")
    consent_revoked: bool = Field(..., description="True if consent was revoked.")
    all_verified_deleted: bool = Field(..., description="True if all biometric data is verified gone.")
    details: str = Field(..., description="Summary of deletion verification.")
