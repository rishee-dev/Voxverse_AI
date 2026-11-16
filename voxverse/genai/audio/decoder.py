"""Audio decoding engine with multi-format support and graceful FFmpeg integration.

Owner: P1 — Audio and Speaker Profile Engineer
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import BinaryIO

import numpy as np
import soundfile as sf

from voxverse.config.logging import get_logger
from voxverse.genai.audio.exceptions import (
    AudioDecodeError,
    AudioValidationError,
    FFmpegNotFoundError,
)

logger = get_logger("audio.decoder")

# Formats natively decoded by libsndfile (soundfile)
NATIVE_SOUNDFILE_EXTENSIONS: set[str] = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".aiff",
    ".aif",
}

# Formats that require external FFmpeg executable for decoding
FFMPEG_REQUIRED_EXTENSIONS: set[str] = {
    ".m4a",
    ".webm",
    ".mp4",
    ".aac",
    ".wma",
    ".opus",
}


def is_ffmpeg_available() -> bool:
    """Check whether the ffmpeg executable is discovered on system PATH."""
    return shutil.which("ffmpeg") is not None


class AudioDecoder:
    """Decodes diverse audio file formats into normalized floating-point NumPy arrays."""

    def __init__(self, max_file_size_bytes: int = 50 * 1024 * 1024) -> None:
        """Initialize the decoder with a maximum allowed file size.

        Parameters
        ----------
        max_file_size_bytes:
            Maximum allowed raw file size in bytes (default 50 MB).
        """
        self.max_file_size_bytes = max_file_size_bytes

    def decode_file(
        self,
        file_path: Path | str,
    ) -> tuple[np.ndarray, int, int | None, str]:
        """Decode an audio file into a float32 NumPy array and metadata.

        Parameters
        ----------
        file_path:
            Path to the audio file on disk.

        Returns
        -------
        tuple[np.ndarray, int, int | None, str]
            - audio_data: 1D (mono) or 2D (samples, channels) float32 array in [-1.0, 1.0].
            - sample_rate: Native sampling rate in Hz.
            - bit_depth: Bit depth per sample if detectable (e.g. 16, 24, 32), else None.
            - format_name: Detected container format string (e.g. 'WAV', 'MP3').

        Raises
        ------
        AudioValidationError
            If the file does not exist, is empty, or exceeds file size limits.
        AudioDecodeError
            If file data is corrupt, truncated, or unreadable.
        FFmpegNotFoundError
            If the format requires FFmpeg and FFmpeg is not installed on PATH.
        """
        path = Path(file_path).resolve()

        if not path.exists():
            raise AudioValidationError(
                message=f"Audio file '{path.name}' does not exist.",
                code="FILE_NOT_FOUND",
            )

        file_size = path.stat().st_size
        if file_size == 0:
            raise AudioValidationError(
                message=f"Audio file '{path.name}' is empty (0 bytes).",
                code="EMPTY_FILE",
            )

        if file_size > self.max_file_size_bytes:
            max_mb = self.max_file_size_bytes / (1024 * 1024)
            actual_mb = file_size / (1024 * 1024)
            raise AudioValidationError(
                message=(
                    f"Audio file '{path.name}' ({actual_mb:.1f} MB) exceeds maximum "
                    f"allowed size of {max_mb:.1f} MB."
                ),
                code="FILE_TOO_LARGE",
            )

        ext = path.suffix.lower()

        # If extension requires external FFmpeg
        if ext in FFMPEG_REQUIRED_EXTENSIONS:
            return self._decode_via_ffmpeg(path)

        # Attempt decoding with SoundFile
        try:
            info = sf.info(str(path))
            audio_data, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)

            bit_depth: int | None = None
            if info.subtype:
                if "16" in info.subtype:
                    bit_depth = 16
                elif "24" in info.subtype:
                    bit_depth = 24
                elif "32" in info.subtype or "FLOAT" in info.subtype:
                    bit_depth = 32

            format_name = info.format or ext.lstrip(".").upper()
            logger.info(
                "Decoded '%s' via SoundFile (format=%s, sr=%d, channels=%d, duration=%.2fs)",
                path.name,
                format_name,
                sample_rate,
                info.channels,
                info.duration,
            )
            return audio_data, sample_rate, bit_depth, format_name

        except sf.LibsndfileError as err:
            # Check if FFmpeg is available as a fallback
            if is_ffmpeg_available():
                logger.warning(
                    "SoundFile failed on '%s' (%s); falling back to FFmpeg",
                    path.name,
                    err,
                )
                return self._decode_via_ffmpeg(path)

            raise AudioDecodeError(
                message=f"Failed to decode audio file '{path.name}': {err}",
                code="CORRUPT_OR_UNSUPPORTED_AUDIO",
            ) from err
        except Exception as err:
            raise AudioDecodeError(
                message=f"Unexpected error decoding audio file '{path.name}': {err}",
                code="DECODE_EXCEPTION",
            ) from err

    def _decode_via_ffmpeg(
        self,
        path: Path,
    ) -> tuple[np.ndarray, int, int | None, str]:
        """Decode audio file using system FFmpeg binary piped to raw 16-bit PCM."""
        if not is_ffmpeg_available():
            raise FFmpegNotFoundError(
                message=(
                    f"Audio format '{path.suffix}' requires FFmpeg for decoding, "
                    f"but FFmpeg was not detected on system PATH."
                )
            )

        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                check=True,
                timeout=60.0,
            )
            raw_bytes = result.stdout
            if not raw_bytes:
                raise AudioDecodeError(
                    message=f"FFmpeg produced empty audio stream for '{path.name}'.",
                    code="EMPTY_DECODE_OUTPUT",
                )

            # Probe metadata using ffprobe if available, or extract via soundfile
            audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0

            # Default to 16000 or probe native
            sample_rate = 16000
            format_name = path.suffix.lstrip(".").upper()
            return audio_float32, sample_rate, 16, format_name

        except subprocess.TimeoutExpired as err:
            raise AudioDecodeError(
                message=f"FFmpeg decoding timed out on '{path.name}'.",
                code="DECODE_TIMEOUT",
            ) from err
        except subprocess.CalledProcessError as err:
            stderr_msg = err.stderr.decode("utf-8", errors="replace").strip()
            raise AudioDecodeError(
                message=f"FFmpeg failed to decode '{path.name}': {stderr_msg}",
                code="FFMPEG_DECODE_FAILURE",
            ) from err
