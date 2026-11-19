"""Input, reference audio, Telugu script, and audio output validation for TTS.

Owner: P3 — Voice Generation Engineer
"""

from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import soundfile as sf

from voxverse.genai.tts.exceptions import (
    InvalidAudioOutputError,
    ReferenceAudioError,
    TeluguScriptValidationError,
)

# Telugu Unicode Block: U+0C00 to U+0C7F
TELUGU_CHAR_PATTERN = re.compile(r"[\u0C00-\u0C7F]")


def validate_telugu_text(text: str, max_characters: int = 500) -> str:
    """Validate that the input text contains Telugu characters and conforms to size limits.

    Parameters
    ----------
    text:
        Raw input string to synthesize.
    max_characters:
        Upper bound on permitted string length.

    Returns
    -------
    str
        Stripped, valid Telugu text.

    Raises
    ------
    TeluguScriptValidationError
        If text is empty, exceeds length, or contains zero Telugu characters.
    """
    cleaned = text.strip() if text else ""
    if not cleaned:
        raise TeluguScriptValidationError(text, "Input text cannot be empty or purely whitespace.")

    if len(cleaned) > max_characters:
        raise TeluguScriptValidationError(
            text,
            f"Input text length of {len(cleaned)} characters exceeds maximum permitted of {max_characters}.",
        )

    if not TELUGU_CHAR_PATTERN.search(cleaned):
        raise TeluguScriptValidationError(
            text,
            "Input text does not contain any characters from the Unicode Telugu script block (U+0C00–U+0C7F).",
        )

    return cleaned


def validate_reference_audio(
    audio_path: str,
    min_duration: float = 2.0,
    max_duration: float = 15.0,
) -> sf.SoundFile:
    """Validate the existence, format, and duration of the speaker reference audio file.

    Parameters
    ----------
    audio_path:
        Path to the reference audio WAV file.
    min_duration:
        Minimum permitted duration in seconds.
    max_duration:
        Maximum permitted duration in seconds.

    Returns
    -------
    sf.SoundFile
        Opened soundfile instance with validated properties.

    Raises
    ------
    ReferenceAudioError
        If audio file is missing, unreadable, or outside duration bounds.
    """
    path_obj = Path(audio_path)
    if not path_obj.exists() or not path_obj.is_file():
        raise ReferenceAudioError(f"Reference audio file does not exist: {audio_path}", path=audio_path)

    try:
        info = sf.info(str(path_obj))
    except Exception as exc:
        raise ReferenceAudioError(
            f"Failed to read audio metadata from '{audio_path}': {exc}",
            path=audio_path,
        ) from exc

    duration = info.duration
    if duration < min_duration:
        raise ReferenceAudioError(
            f"Reference audio duration of {duration:.2f}s is shorter than minimum required {min_duration:.1f}s.",
            path=audio_path,
        )

    if duration > max_duration:
        raise ReferenceAudioError(
            f"Reference audio duration of {duration:.2f}s exceeds maximum permitted {max_duration:.1f}s.",
            path=audio_path,
        )

    if info.samplerate < 8000:
        raise ReferenceAudioError(
            f"Reference audio sample rate ({info.samplerate} Hz) is too low for acoustic conditioning.",
            path=audio_path,
        )

    return info


def validate_audio_output(
    audio_data: np.ndarray,
    sample_rate: int,
    output_path: str | None = None,
) -> None:
    """Validate synthesized waveform array for numerical integrity, non-emptiness, and range.

    Parameters
    ----------
    audio_data:
        Numpy float array representing synthesized PCM samples.
    sample_rate:
        Sample rate in Hz.
    output_path:
        Optional file path where audio was written.

    Raises
    ------
    InvalidAudioOutputError
        If audio is empty, contains NaN/Inf, or has zero duration.
    """
    if audio_data is None or len(audio_data) == 0:
        raise InvalidAudioOutputError("Synthesized audio buffer is empty.", path=output_path)

    if not np.isfinite(audio_data).all():
        raise InvalidAudioOutputError(
            "Synthesized audio contains invalid numerical values (NaN or Infinity).",
            path=output_path,
        )

    duration = len(audio_data) / sample_rate
    if duration < 0.1:
        raise InvalidAudioOutputError(
            f"Synthesized audio duration of {duration:.3f}s is below minimum audible threshold.",
            path=output_path,
        )
