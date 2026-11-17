"""Domain exceptions for speaker enrollment, consent verification, and profile management.

Owner: P1 — Audio and Speaker Profile Engineer
"""

from __future__ import annotations


class SpeakerError(Exception):
    """Base exception for all speaker profile and enrollment failures."""

    def __init__(self, message: str, code: str = "SPEAKER_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class ConsentRequiredError(SpeakerError):
    """Raised when an operation is attempted without affirmative speaker consent."""

    def __init__(
        self,
        message: str = "Explicit affirmative consent is strictly required for speaker enrollment.",
        code: str = "CONSENT_REQUIRED",
    ) -> None:
        super().__init__(message, code=code)


class ConsentRevokedError(SpeakerError):
    """Raised when attempting to access or process a profile whose consent has been revoked."""

    def __init__(
        self,
        message: str = "Consent for this speaker profile has been revoked. The profile is unusable.",
        code: str = "CONSENT_REVOKED",
    ) -> None:
        super().__init__(message, code=code)


class InvalidConsentError(SpeakerError):
    """Raised when consent declaration is malformed, missing required purposes, or lacks ownership confirmation."""

    def __init__(self, message: str, code: str = "INVALID_CONSENT") -> None:
        super().__init__(message, code=code)


class UnauthorizedProfileAccessError(SpeakerError):
    """Raised when an authenticated owner attempts to access or delete another user's profile."""

    def __init__(
        self,
        message: str = "You do not have authorization to access or modify this voice profile.",
        code: str = "UNAUTHORIZED_PROFILE_ACCESS",
    ) -> None:
        super().__init__(message, code=code)


class ProfileNotFoundError(SpeakerError):
    """Raised when a requested voice profile does not exist or has been deleted."""

    def __init__(
        self,
        message: str = "Voice profile not found.",
        code: str = "PROFILE_NOT_FOUND",
    ) -> None:
        super().__init__(message, code=code)


class EnrollmentValidationError(SpeakerError):
    """Raised when enrollment audio violates duration, acoustic, or speech activity requirements."""

    def __init__(
        self,
        message: str,
        code: str = "ENROLLMENT_VALIDATION_FAILED",
        issues: list[str] | None = None,
    ) -> None:
        super().__init__(message, code=code)
        self.issues = issues or [message]


class ProfileDeletionError(SpeakerError):
    """Raised when profile or biometric data deletion cannot be verified on disk."""

    def __init__(self, message: str, code: str = "PROFILE_DELETION_FAILED") -> None:
        super().__init__(message, code=code)


class PathTraversalError(SpeakerError):
    """Raised when an identifier or filename attempts directory traversal outside storage."""

    def __init__(
        self,
        message: str = "Path traversal or illegal identifier detected.",
        code: str = "PATH_TRAVERSAL_DETECTED",
    ) -> None:
        super().__init__(message, code=code)
