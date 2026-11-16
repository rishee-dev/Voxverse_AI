"""Audio normalizer producing standardized 16 kHz mono 16-bit PCM WAV streams.

Owner: P1 — Audio and Speaker Profile Engineer
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import scipy.signal as signal
import soundfile as sf

from voxverse.config.logging import get_logger
from voxverse.genai.audio.decoder import AudioDecoder
from voxverse.genai.audio.exceptions import AudioProcessingError
from voxverse.genai.audio.schemas import AudioMetadata, AudioQualityReport
from voxverse.genai.audio.validator import AudioValidator

logger = get_logger("audio.normalizer")

TARGET_SAMPLE_RATE: int = 16000
TARGET_CHANNELS: int = 1
TARGET_SUBTYPE: str = "PCM_16"
TARGET_FORMAT: str = "WAV"


class AudioNormalizer:
    """Converts diverse audio formats into standardized 16 kHz mono 16-bit PCM WAV files.

    Invariants:
    1. The original input file is NEVER modified in-place.
    2. Conversion is strictly deterministic.
    3. Output is guaranteed to be 16,000 Hz, 1 channel, 16-bit linear PCM WAV.
    """

    def __init__(
        self,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        decoder: AudioDecoder | None = None,
        validator: AudioValidator | None = None,
    ) -> None:
        self.target_sample_rate = target_sample_rate
        self.decoder = decoder or AudioDecoder()
        self.validator = validator or AudioValidator()

    def normalize(
        self,
        input_path: Path | str,
        output_path: Path | str,
        *,
        validate_input: bool = True,
    ) -> tuple[Path, AudioMetadata, AudioQualityReport]:
        """Normalize an audio file to standardized 16 kHz mono 16-bit PCM WAV.

        Parameters
        ----------
        input_path:
            Path to the source audio file.
        output_path:
            Target destination path for the normalized .wav file.
        validate_input:
            Whether to run quality validation on the input.

        Returns
        -------
        tuple[Path, AudioMetadata, AudioQualityReport]
            - Destination Path of the normalized audio file.
            - AudioMetadata describing original and normalized attributes.
            - AudioQualityReport documenting quality metrics.

        Raises
        ------
        AudioValidationError
            If the audio violates size, channel, or quality limits.
        AudioDecodeError
            If decoding fails.
        """
        source = Path(input_path).resolve()
        target = Path(output_path).resolve()

        if source == target:
            raise AudioProcessingError(
                "Source and target paths cannot be identical. In-place modification is forbidden.",
                code="IN_PLACE_MUTATION_FORBIDDEN",
            )

        # 1. Decode original audio
        audio_data, original_sr, bit_depth, format_name = self.decoder.decode_file(source)

        original_channels = 1 if audio_data.ndim == 1 else audio_data.shape[1]
        original_frames = audio_data.shape[0] if audio_data.ndim >= 1 else 0
        duration_seconds = original_frames / float(original_sr) if original_sr > 0 else 0.0
        file_size_bytes = source.stat().st_size

        # 2. Validate input quality
        quality_report = self.validator.validate(audio_data, original_sr, strict=False)

        # 3. Channel downmixing to mono
        if audio_data.ndim > 1 and audio_data.shape[1] > 1:
            # Average across channels
            mono_audio = np.mean(audio_data, axis=1, dtype=np.float32)
            logger.info("Downmixed %d channels to mono", audio_data.shape[1])
        else:
            mono_audio = audio_data.flatten().astype(np.float32)

        # 4. Resample to target sample rate (16,000 Hz)
        if original_sr != self.target_sample_rate:
            gcd = math.gcd(self.target_sample_rate, original_sr)
            up = self.target_sample_rate // gcd
            down = original_sr // gcd
            resampled_audio = signal.resample_poly(mono_audio, up, down).astype(np.float32)
            logger.info(
                "Resampled audio from %d Hz to %d Hz (factor %d/%d, frames: %d -> %d)",
                original_sr,
                self.target_sample_rate,
                up,
                down,
                len(mono_audio),
                len(resampled_audio),
            )
        else:
            resampled_audio = mono_audio

        # 5. Prevent clipping on output conversion
        peak = float(np.max(np.abs(resampled_audio))) if resampled_audio.size > 0 else 0.0
        if peak > 1.0:
            logger.warning("Resampled peak (%.4f) exceeds 1.0; applying headroom scaling", peak)
            resampled_audio = resampled_audio / (peak * 1.001)

        # 6. Ensure destination directory exists and write 16-bit PCM WAV
        target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            str(target),
            resampled_audio,
            samplerate=self.target_sample_rate,
            subtype=TARGET_SUBTYPE,
            format=TARGET_FORMAT,
        )

        # 7. Construct complete metadata model
        metadata = AudioMetadata(
            original_format=format_name,
            sample_rate=original_sr,
            channels=original_channels,
            bit_depth=bit_depth,
            duration_seconds=duration_seconds,
            frame_count=original_frames,
            file_size_bytes=file_size_bytes,
            normalized_sample_rate=self.target_sample_rate,
            normalized_channels=TARGET_CHANNELS,
            normalized_format=TARGET_FORMAT,
        )

        logger.info(
            "Normalized '%s' -> '%s' (duration=%.2fs, output_frames=%d)",
            source.name,
            target.name,
            metadata.duration_seconds,
            len(resampled_audio),
        )

        return target, metadata, quality_report
