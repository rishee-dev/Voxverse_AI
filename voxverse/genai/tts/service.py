"""High-level TTS orchestration service integrating voice profile storage and synthesis.

Owner: P3 — Voice Generation Engineer with P4 — Pipeline Engineer
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from voxverse.genai.speaker.schemas import ConsentStatus, VoiceProfile
from voxverse.genai.tts.base import BaseTTSEngine
from voxverse.genai.tts.exceptions import (
    MissingConsentError,
    MissingReferenceTranscriptError,
    ReferenceAudioError,
    VoiceProfileNotFoundError,
)
from voxverse.genai.tts.schemas import (
    SpeakerConditioningData,
    TTSRequest,
    TTSResult,
)
from voxverse.storage.voice_storage import VoiceProfileStorage

logger = logging.getLogger(__name__)


class TTSService:
    """Service orchestrating voice profile retrieval, consent validation, and speech synthesis."""

    def __init__(
        self,
        engine: BaseTTSEngine,
        voice_storage: VoiceProfileStorage | None = None,
    ) -> None:
        self.engine = engine
        self.voice_storage = voice_storage

    def synthesize(
        self,
        request: TTSRequest,
        *,
        reference_transcript: str | None = None,
        output_path: str | None = None,
    ) -> TTSResult:
        """Process a TTS synthesis request with optional speaker conditioning.

        Parameters
        ----------
        request:
            Validated synthesis request payload.
        reference_transcript:
            Optional verbatim transcript of the reference audio if not stored on disk.
        output_path:
            Optional explicit destination path for the generated audio file.

        Returns
        -------
        TTSResult
            Synthesis output including file path, duration, telemetry, and similarity metrics.
        """
        conditioning_data: SpeakerConditioningData | None = None

        if request.voice_id is not None:
            if self.voice_storage is None:
                raise VoiceProfileNotFoundError(request.voice_id)

            profile = self.voice_storage.get_profile(request.voice_id)
            if profile is None:
                raise VoiceProfileNotFoundError(request.voice_id)

            # Affirmative consent verification (P1/P3 security boundary)
            if profile.consent.status != ConsentStatus.GRANTED:
                raise MissingConsentError(
                    request.voice_id,
                    f"Affirmative speaker consent is revoked or not granted for owner '{profile.owner_id}'.",
                )

            # Reference audio existence check
            try:
                ref_path = self.voice_storage.get_reference_audio_path(request.voice_id)
            except Exception as exc:
                raise ReferenceAudioError(f"Could not retrieve reference audio for profile '{request.voice_id}': {exc}") from exc

            if not ref_path or not ref_path.exists():
                raise ReferenceAudioError(
                    f"Reference audio file is missing for voice profile '{request.voice_id}'.",
                    path=str(ref_path) if ref_path else None,
                )

            # Check for transcript: either explicitly provided or in reference_transcript.txt
            transcript = reference_transcript
            if not transcript:
                transcript_file = ref_path.parent / "transcript.txt"
                if transcript_file.exists():
                    transcript = transcript_file.read_text(encoding="utf-8").strip()

            if not transcript or not transcript.strip():
                raise MissingReferenceTranscriptError(request.voice_id)

            conditioning_data = SpeakerConditioningData(
                voice_id=profile.profile_id,
                owner_id=profile.owner_id,
                speaker_name=profile.display_name,
                reference_audio_path=str(ref_path),
                reference_transcript=transcript,
                consent_hash=profile.consent.consent_id,
                sample_rate=16000,
            )

        logger.info("Submitting text synthesis to %s engine...", self.engine.model_name)
        return self.engine.synthesize(
            text=request.text,
            conditioning_data=conditioning_data,
            nfe_steps=request.nfe_steps,
            speed=request.speed,
            output_path=output_path,
        )
