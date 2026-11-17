"""VoxVerse GenAI Speaker — Voice enrollment, affirmative consent, and profile lifecycle (P1).

Owner: P1 — Audio and Speaker Profile Engineer
"""

from voxverse.genai.speaker.exceptions import (
    ConsentRequiredError,
    ConsentRevokedError,
    EnrollmentValidationError,
    InvalidConsentError,
    PathTraversalError,
    ProfileDeletionError,
    ProfileNotFoundError,
    SpeakerError,
    UnauthorizedProfileAccessError,
)
from voxverse.genai.speaker.schemas import (
    ConsentPurpose,
    ConsentRecord,
    ConsentStatus,
    DeletionResult,
    EnrollmentStatus,
    EnrollmentValidationReport,
    VoiceEnrollmentRequest,
    VoiceProfile,
    VoiceProfileMetadata,
)
from voxverse.genai.speaker.service import SpeakerProfileService
from voxverse.genai.speaker.validator import EnrollmentValidator

__all__ = [
    "ConsentPurpose",
    "ConsentRecord",
    "ConsentRequiredError",
    "ConsentRevokedError",
    "ConsentStatus",
    "DeletionResult",
    "EnrollmentStatus",
    "EnrollmentValidationError",
    "EnrollmentValidationReport",
    "EnrollmentValidator",
    "InvalidConsentError",
    "PathTraversalError",
    "ProfileDeletionError",
    "ProfileNotFoundError",
    "SpeakerError",
    "SpeakerProfileService",
    "UnauthorizedProfileAccessError",
    "VoiceEnrollmentRequest",
    "VoiceProfile",
    "VoiceProfileMetadata",
]
