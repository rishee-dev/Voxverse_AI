"""Audio quality validation and integrity analysis.

Owner: P1 — Audio and Speaker Profile Engineer
"""

from __future__ import annotations

import numpy as np

from voxverse.config.logging import get_logger
from voxverse.genai.audio.exceptions import AudioValidationError
from voxverse.genai.audio.schemas import AudioQualityReport

logger = get_logger("audio.validator")

# Default boundary parameters
DEFAULT_MIN_DURATION_SECONDS: float = 0.5
DEFAULT_MAX_DURATION_SECONDS: float = 300.0  # 5 minutes
DEFAULT_MIN_SAMPLE_RATE: int = 8000
DEFAULT_MAX_SAMPLE_RATE: int = 192000
DEFAULT_MAX_CHANNELS: int = 8
DEFAULT_SILENCE_RMS_THRESHOLD: float = 0.001  # Approximately -60 dBFS
DEFAULT_CLIPPING_PEAK_THRESHOLD: float = 0.999
DEFAULT_CLIPPING_RATIO_THRESHOLD: float = 0.001  # 0.1% of samples at ceiling


class AudioValidator:
    """Measures audio physical attributes and validates quality standards."""

    def __init__(
        self,
        min_duration_seconds: float = DEFAULT_MIN_DURATION_SECONDS,
        max_duration_seconds: float = DEFAULT_MAX_DURATION_SECONDS,
        min_sample_rate: int = DEFAULT_MIN_SAMPLE_RATE,
        max_sample_rate: int = DEFAULT_MAX_SAMPLE_RATE,
        max_channels: int = DEFAULT_MAX_CHANNELS,
        silence_rms_threshold: float = DEFAULT_SILENCE_RMS_THRESHOLD,
        clipping_peak_threshold: float = DEFAULT_CLIPPING_PEAK_THRESHOLD,
    ) -> None:
        self.min_duration_seconds = min_duration_seconds
        self.max_duration_seconds = max_duration_seconds
        self.min_sample_rate = min_sample_rate
        self.max_sample_rate = max_sample_rate
        self.max_channels = max_channels
        self.silence_rms_threshold = silence_rms_threshold
        self.clipping_peak_threshold = clipping_peak_threshold

    def validate(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        *,
        strict: bool = False,
    ) -> AudioQualityReport:
        """Analyze audio array quality and return a structured report.

        Parameters
        ----------
        audio_data:
            NumPy array of audio samples (1D or 2D).
        sample_rate:
            Audio sampling rate in Hertz.
        strict:
            If True, immediately raises AudioValidationError on quality failure.

        Returns
        -------
        AudioQualityReport
            Structured assessment containing RMS, peak amplitude, duration, and issues.

        Raises
        ------
        AudioValidationError
            If strict is True and validation detects critical errors.
        """
        issues: list[str] = []

        # 1. Check array dimension and sample count
        if audio_data is None or audio_data.size == 0:
            msg = "Audio array contains zero samples."
            issues.append(msg)
            if strict:
                raise AudioValidationError(msg, code="EMPTY_AUDIO", issues=issues)
            return AudioQualityReport(
                is_valid=False,
                rms_amplitude=0.0,
                peak_amplitude=0.0,
                is_silent=True,
                is_clipped=False,
                duration_seconds=0.0,
                issues=issues,
            )

        # 2. Check channel configuration
        channels = 1 if audio_data.ndim == 1 else audio_data.shape[1]
        if channels < 1 or channels > self.max_channels:
            msg = f"Invalid channel count {channels} (allowed: 1 to {self.max_channels})."
            issues.append(msg)

        # 3. Check sample rate
        if sample_rate < self.min_sample_rate or sample_rate > self.max_sample_rate:
            msg = (
                f"Sample rate {sample_rate} Hz is outside valid bounds "
                f"[{self.min_sample_rate}, {self.max_sample_rate}] Hz."
            )
            issues.append(msg)

        # 4. Check duration
        num_frames = audio_data.shape[0] if audio_data.ndim >= 1 else 0
        duration_seconds = num_frames / float(sample_rate) if sample_rate > 0 else 0.0

        if duration_seconds < self.min_duration_seconds:
            msg = (
                f"Audio duration ({duration_seconds:.2f}s) is shorter than "
                f"minimum threshold ({self.min_duration_seconds:.2f}s)."
            )
            issues.append(msg)

        if duration_seconds > self.max_duration_seconds:
            msg = (
                f"Audio duration ({duration_seconds:.2f}s) exceeds maximum "
                f"allowed threshold ({self.max_duration_seconds:.2f}s)."
            )
            issues.append(msg)

        # 5. Measure amplitude metrics
        flat_samples = audio_data.flatten()
        peak_amplitude = float(np.max(np.abs(flat_samples)))
        rms_amplitude = float(np.sqrt(np.mean(flat_samples**2)))

        # 6. Silence detection
        is_silent = rms_amplitude < self.silence_rms_threshold
        if is_silent:
            msg = (
                f"Audio is predominantly silent (RMS: {rms_amplitude:.6f} < "
                f"threshold {self.silence_rms_threshold:.6f})."
            )
            issues.append(msg)

        # 7. Clipping detection
        clipped_samples = np.sum(np.abs(flat_samples) >= self.clipping_peak_threshold)
        clipping_ratio = float(clipped_samples) / float(flat_samples.size)
        is_clipped = peak_amplitude >= 1.0 or clipping_ratio > DEFAULT_CLIPPING_RATIO_THRESHOLD

        if is_clipped:
            msg = (
                f"Audio contains clipped samples (Peak: {peak_amplitude:.4f}, "
                f"Clipped ratio: {clipping_ratio*100:.2f}%)."
            )
            issues.append(msg)

        is_valid = len(issues) == 0

        if strict and not is_valid:
            raise AudioValidationError(
                message=f"Audio validation failed: {'; '.join(issues)}",
                code="AUDIO_QUALITY_FAILURE",
                issues=issues,
            )

        report = AudioQualityReport(
            is_valid=is_valid,
            rms_amplitude=rms_amplitude,
            peak_amplitude=peak_amplitude,
            is_silent=is_silent,
            is_clipped=is_clipped,
            duration_seconds=duration_seconds,
            issues=issues,
        )

        logger.info(
            "Validation completed: valid=%s, duration=%.2fs, rms=%.4f, peak=%.4f, issues=%d",
            report.is_valid,
            report.duration_seconds,
            report.rms_amplitude,
            report.peak_amplitude,
            len(report.issues),
        )

        return report
