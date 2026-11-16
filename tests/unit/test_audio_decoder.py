"""Unit tests for the audio decoder.

Covers:
- Valid mono WAV decoding
- Stereo WAV decoding
- Multi-format decoding (MP3, FLAC, OGG)
- Non-existent file rejection
- Empty file rejection
- Corrupted file handling
- Unsupported container format (M4A/WebM) FFmpeg requirement behavior
- Maximum file size enforcement

Owner: P1 — Audio and Speaker Profile Engineer
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voxverse.genai.audio.decoder import AudioDecoder, is_ffmpeg_available
from voxverse.genai.audio.exceptions import (
    AudioDecodeError,
    AudioValidationError,
    FFmpegNotFoundError,
)


@pytest.fixture
def decoder() -> AudioDecoder:
    """Return a standard AudioDecoder instance."""
    return AudioDecoder(max_file_size_bytes=5 * 1024 * 1024)  # 5 MB limit for tests


def _create_synthetic_wav(
    path: Path,
    duration: float = 1.0,
    sample_rate: int = 16000,
    channels: int = 1,
    frequency: float = 440.0,
) -> Path:
    """Helper to generate a deterministic synthetic WAV file."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float32)
    tone = 0.5 * np.sin(2 * np.pi * frequency * t)
    if channels == 1:
        data = tone
    else:
        data = np.column_stack([tone] * channels)

    sf.write(str(path), data, sample_rate, subtype="PCM_16", format="WAV")
    return path


def test_decode_valid_mono_wav(decoder: AudioDecoder, tmp_path: Path) -> None:
    """Test decoding a standard mono 16 kHz WAV file."""
    wav_path = _create_synthetic_wav(tmp_path / "mono.wav", duration=1.0, sample_rate=16000, channels=1)

    audio, sr, bit_depth, fmt = decoder.decode_file(wav_path)

    assert sr == 16000
    assert audio.ndim == 1
    assert len(audio) == 16000
    assert fmt == "WAV"
    assert bit_depth == 16


def test_decode_valid_stereo_wav(decoder: AudioDecoder, tmp_path: Path) -> None:
    """Test decoding a stereo 44.1 kHz WAV file."""
    wav_path = _create_synthetic_wav(tmp_path / "stereo.wav", duration=0.5, sample_rate=44100, channels=2)

    audio, sr, bit_depth, fmt = decoder.decode_file(wav_path)

    assert sr == 44100
    assert audio.ndim == 2
    assert audio.shape[1] == 2
    assert audio.shape[0] == int(44100 * 0.5)
    assert fmt == "WAV"


def test_decode_native_flac(decoder: AudioDecoder, tmp_path: Path) -> None:
    """Test decoding a lossless FLAC file via native SoundFile backend."""
    flac_path = tmp_path / "test.flac"
    t = np.linspace(0, 0.5, 8000, endpoint=False, dtype=np.float32)
    sf.write(str(flac_path), 0.4 * np.sin(2 * np.pi * 300 * t), 16000, format="FLAC")

    audio, sr, bit_depth, fmt = decoder.decode_file(flac_path)

    assert sr == 16000
    assert fmt == "FLAC"
    assert len(audio) == 8000


def test_decode_native_ogg(decoder: AudioDecoder, tmp_path: Path) -> None:
    """Test decoding an OGG Vorbis file via native SoundFile backend."""
    ogg_path = tmp_path / "test.ogg"
    t = np.linspace(0, 0.5, 8000, endpoint=False, dtype=np.float32)
    sf.write(str(ogg_path), 0.4 * np.sin(2 * np.pi * 300 * t), 16000, format="OGG")

    audio, sr, _, fmt = decoder.decode_file(ogg_path)

    assert sr == 16000
    assert fmt == "OGG"
    assert len(audio) == 8000


def test_decode_native_mp3(decoder: AudioDecoder, tmp_path: Path) -> None:
    """Test decoding an MP3 file via native SoundFile backend."""
    mp3_path = tmp_path / "test.mp3"
    t = np.linspace(0, 0.5, 8000, endpoint=False, dtype=np.float32)
    sf.write(str(mp3_path), 0.4 * np.sin(2 * np.pi * 300 * t), 16000, format="MP3")

    audio, sr, _, fmt = decoder.decode_file(mp3_path)

    assert sr == 16000
    assert fmt == "MP3"
    assert len(audio) > 0


def test_decode_non_existent_file(decoder: AudioDecoder, tmp_path: Path) -> None:
    """Test that a non-existent file path raises AudioValidationError."""
    missing_path = tmp_path / "non_existent.wav"

    with pytest.raises(AudioValidationError) as exc_info:
        decoder.decode_file(missing_path)

    assert exc_info.value.code == "FILE_NOT_FOUND"


def test_decode_empty_file(decoder: AudioDecoder, tmp_path: Path) -> None:
    """Test that a zero-byte file raises AudioValidationError."""
    empty_path = tmp_path / "empty.wav"
    empty_path.write_bytes(b"")

    with pytest.raises(AudioValidationError) as exc_info:
        decoder.decode_file(empty_path)

    assert exc_info.value.code == "EMPTY_FILE"


def test_decode_corrupted_file(decoder: AudioDecoder, tmp_path: Path) -> None:
    """Test that arbitrary garbage data with a .wav extension raises AudioDecodeError."""
    corrupt_path = tmp_path / "corrupt.wav"
    corrupt_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt invalid_stream_data_payload_123456789")

    with pytest.raises(AudioDecodeError):
        decoder.decode_file(corrupt_path)


def test_decode_file_exceeding_max_size(tmp_path: Path) -> None:
    """Test that exceeding the configured max file size raises AudioValidationError."""
    small_decoder = AudioDecoder(max_file_size_bytes=100)  # 100 bytes limit
    large_file = tmp_path / "large.wav"
    large_file.write_bytes(b"x" * 200)

    with pytest.raises(AudioValidationError) as exc_info:
        small_decoder.decode_file(large_file)

    assert exc_info.value.code == "FILE_TOO_LARGE"


def test_decode_unsupported_m4a_behavior(decoder: AudioDecoder, tmp_path: Path) -> None:
    """Test that requesting an M4A file either decodes via FFmpeg or raises FFmpegNotFoundError."""
    m4a_path = tmp_path / "sample.m4a"
    m4a_path.write_bytes(b"dummy m4a payload")

    if not is_ffmpeg_available():
        with pytest.raises(FFmpegNotFoundError):
            decoder.decode_file(m4a_path)
    else:
        # If FFmpeg was installed, it would attempt FFmpeg decode and raise decode error on bad data
        with pytest.raises(AudioDecodeError):
            decoder.decode_file(m4a_path)
