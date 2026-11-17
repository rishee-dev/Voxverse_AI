"""Enrollment audio validator reusing P1 AudioDecoder, AudioValidator, and SileroVADDetector.

Owner: P1 — Audio and Speaker Profile Engineer
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from voxverse.config.logging import get_logger
from voxverse.config.settings import Settings, get_settings
from voxverse.genai.audio.decoder import AudioDecoder
from voxverse.genai.audio.vad import SileroVADDetector
from voxverse.genai.audio.validator import AudioValidator
from voxverse.genai.speaker.exceptions import EnrollmentValidationError
from voxverse.genai.speaker.schemas import EnrollmentValidationReport

logger = get_logger("speaker.validator")


class EnrollmentValidator:
    """Acoustic and speech activity validator for speaker enrollment audio.

    Engineering Invariants:
    1. Reuses P1's AudioDecoder, AudioValidator, and SileroVADDetector without duplication.
    2. Enforces minimum total audio duration (default: 3.0s) and maximum duration (default: 60.0s).
    3. Enforces measurable human speech presence via Silero VAD (minimum speech: 1.5s).
    4. Rejects silent, clipped, or speech-free recordings before storage.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        decoder: AudioDecoder | None = None,
        validator: AudioValidator | None = None,
        vad_detector: SileroVADDetector | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.decoder = decoder or AudioDecoder()
        self.validator = validator or AudioValidator()
        self.vad_detector = vad_detector or SileroVADDetector()

    def validate_enrollment_audio(
        self,
        audio_path: Path | str,
        *,
        min_duration: float | None = None,
        max_duration: float | None = None,
        min_speech_duration: float | None = None,
    ) -> tuple[np.ndarray, int, EnrollmentValidationReport]:
        """Validate enrollment audio for format, duration, acoustic quality, and speech presence.

        Returns
        -------
        tuple[np.ndarray, int, EnrollmentValidationReport]
            - Decoded audio float32 numpy array.
            - Original sample rate.
            - Detailed EnrollmentValidationReport.

        Raises
        ------
        EnrollmentValidationError
            If any validation constraint is violated.
        """
        path = Path(audio_path).resolve()
        min_dur = min_duration or self.settings.min_enrollment_duration_seconds
        max_dur = max_duration or self.settings.max_enrollment_duration_seconds
        min_speech = min_speech_duration or self.settings.min_speech_duration_seconds

        logger.info("Validating enrollment audio: %s", path.name)

        # 1. Decode audio
        try:
            audio_data, original_sr, bit_depth, format_name = self.decoder.decode_file(path)
        except Exception as e:
            logger.error("Enrollment audio decode failed for %s: %s", path.name, e)
            raise EnrollmentValidationError(
                f"Enrollment audio file could not be decoded: {e}",
                code="ENROLLMENT_DECODE_FAILED",
            ) from e

        frames = audio_data.shape[0] if audio_data.ndim >= 1 else 0
        channels = 1 if audio_data.ndim == 1 else audio_data.shape[1]
        duration = frames / float(original_sr) if original_sr > 0 else 0.0

        issues: list[str] = []

        # 2. Duration Bounds Check
        if duration < min_dur:
            issues.append(
                f"Audio duration {duration:.2f}s is below minimum enrollment requirement of {min_dur:.1f}s."
            )
        if duration > max_dur:
            issues.append(
                f"Audio duration {duration:.2f}s exceeds maximum enrollment limit of {max_dur:.1f}s."
            )

        # 3. Acoustic Quality Assessment (RMS, Peak, Clipping, Silence)
        quality = self.validator.validate(audio_data, original_sr, strict=False)
        if quality.is_silent:
            issues.append("Enrollment audio is classified as digital silence (RMS < 0.001).")
        if quality.is_clipped:
            issues.append("Enrollment audio exhibits severe digital clipping / saturation (Peak >= 0.999).")

        # 4. Human Speech Activity Validation via Silero VAD
        # Ensure 1D mono for VAD
        mono_audio = np.mean(audio_data, axis=1, dtype=np.float32) if audio_data.ndim > 1 else audio_data.astype(np.float32)

        # Resample for VAD if not 16 kHz
        if original_sr != 16000:
            import scipy.signal as signal
            gcd = np.gcd(16000, original_sr)
            up = 16000 // gcd
            down = original_sr // gcd
            vad_input = signal.resample_poly(mono_audio, up, down).astype(np.float32)
        else:
            vad_input = mono_audio

        vad_result = self.vad_detector.detect_speech(vad_input, sample_rate=16000)
        speech_duration = sum((seg.end_ms - seg.start_ms) for seg in vad_result.segments) / 1000.0
        speech_ratio = speech_duration / duration if duration > 0 else 0.0

        if speech_duration == 0.0:
            issues.append(
                "No detectable human speech found in enrollment recording. "
                "Silero VAD classified the entire audio as non-speech."
            )
        elif speech_duration < min_speech:
            issues.append(
                f"Detected speech duration {speech_duration:.2f}s is insufficient. "
                f"A minimum of {min_speech:.1f}s of active human speech is required."
            )

        is_valid = len(issues) == 0

        report = EnrollmentValidationReport(
            is_valid=is_valid,
            duration_seconds=round(duration, 3),
            sample_rate=original_sr,
            channels=channels,
            rms_amplitude=round(quality.rms_amplitude, 5),
            peak_amplitude=round(quality.peak_amplitude, 5),
            is_silent=quality.is_silent,
            is_clipped=quality.is_clipped,
            detected_speech_duration_seconds=round(speech_duration, 3),
            speech_ratio=round(speech_ratio, 4),
            issues=issues,
        )

        if not is_valid:
            logger.warning("Enrollment validation failed for %s: %s", path.name, issues)
            raise EnrollmentValidationError(
                f"Enrollment audio rejected: {'; '.join(issues)}",
                code="ENROLLMENT_VALIDATION_FAILED",
                issues=issues,
            )

        logger.info(
            "Enrollment validation passed for %s: duration=%.2fs, speech=%.2fs (%.1f%%)",
            path.name,
            duration,
            speech_duration,
            speech_ratio * 100,
        )
        return audio_data, original_sr, report
