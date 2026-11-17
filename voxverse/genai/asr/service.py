"""High-level ASR service coordinating transcription and English language validation.

Owner: P2 — Speech Understanding Engineer
"""

from __future__ import annotations

from pathlib import Path

from voxverse.config.logging import get_logger
from voxverse.genai.asr.base import BaseASREngine
from voxverse.genai.asr.exceptions import NonEnglishAudioError
from voxverse.genai.asr.faster_whisper_engine import FasterWhisperEngine
from voxverse.genai.asr.schemas import (
    LanguageValidationResult,
    TranscriptionResult,
)

logger = get_logger("asr.service")


class ASRService:
    """Service orchestrating speech-to-text inference and English validation.

    Adheres to dependency injection: accepts any BaseASREngine implementation.
    """

    def __init__(self, engine: BaseASREngine | None = None) -> None:
        self.engine = engine or FasterWhisperEngine()

    def transcribe_file(
        self,
        audio_path: Path | str,
        *,
        validate_english: bool = True,
        beam_size: int = 5,
        temperature: float = 0.0,
    ) -> TranscriptionResult:
        """Transcribe an audio file with optional English language validation.

        Parameters
        ----------
        audio_path:
            Path to the input audio file (typically normalized 16 kHz mono WAV from P1).
        validate_english:
            If True, runs language detection (if supported) and rejects non-English speech.
        beam_size:
            Beam search width.
        temperature:
            Decoding temperature.

        Returns
        -------
        TranscriptionResult
            Complete transcription output including segments, timing, RTF, and validation metadata.

        Raises
        ------
        NonEnglishAudioError
            If audio is classified as non-English and validate_english is True.
        AudioInputError
            If audio file is unreadable, missing, or empty.
        TranscriptionError
            If inference fails.
        """
        path = Path(audio_path).resolve()
        logger.info("ASRService: Processing transcription for %s", path.name)

        lang_val: LanguageValidationResult | None = None

        # 1. Optional English language validation
        if validate_english:
            lang_val = self.engine.detect_language(path)
            logger.debug(
                "Language detection for %s: lang=%s, prob=%s, status=%s",
                path.name,
                lang_val.detected_language,
                lang_val.language_probability,
                lang_val.status,
            )
            if not lang_val.is_english and lang_val.status == "REJECTED_NON_ENGLISH":
                logger.warning(
                    "Non-English audio rejected for %s: detected %s (%s)",
                    path.name,
                    lang_val.detected_language,
                    lang_val.reason,
                )
                raise NonEnglishAudioError(
                    message=f"Non-English audio rejected: {lang_val.reason}",
                    detected_language=lang_val.detected_language,
                    probability=lang_val.language_probability,
                )

        # 2. Execute speech transcription
        result = self.engine.transcribe(
            path,
            language="en",
            beam_size=beam_size,
            temperature=temperature,
        )

        # 3. Attach validation result if generated
        if lang_val is not None:
            result.language_validation = lang_val

        logger.info(
            "Transcription completed for %s: text_len=%d, RTF=%.4f, peak_ram=%.2fMB",
            path.name,
            len(result.text),
            result.real_time_factor,
            result.peak_memory_mb or 0.0,
        )
        return result
