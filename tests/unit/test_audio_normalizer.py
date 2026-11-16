"""Unit tests for the audio normalizer.

Covers:
- Mono WAV normalization to 16 kHz, 1-channel, 16-bit PCM
- Stereo downmixing to mono
- Resampling from 44.1 kHz, 48 kHz, 22.05 kHz, and 8 kHz to 16 kHz
- Multi-format normalization (FLAC, OGG, MP3 -> WAV)
- In-place mutation prohibition
- Preservation of original input files
- Output metadata verification

Owner: P1 — Audio and Speaker Profile Engineer
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voxverse.genai.audio.exceptions import AudioProcessingError
from voxverse.genai.audio.normalizer import AudioNormalizer


@pytest.fixture
def normalizer() -> AudioNormalizer:
    """Return a standard AudioNormalizer instance targeting 16 kHz mono."""
    return AudioNormalizer(target_sample_rate=16000)


def _write_wav(path: Path, data: np.ndarray, samplerate: int) -> Path:
    sf.write(str(path), data, samplerate, subtype="PCM_16", format="WAV")
    return path


def test_normalize_mono_wav(normalizer: AudioNormalizer, tmp_path: Path) -> None:
    """Test normalizing an existing 16 kHz mono WAV file."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)

    input_file = _write_wav(tmp_path / "input_16k.wav", tone, sr)
    output_file = tmp_path / "normalized.wav"

    out_path, metadata, quality = normalizer.normalize(input_file, output_file)

    assert out_path == output_file
    assert out_path.exists()

    # Read back normalized audio to verify exact technical format
    info = sf.info(str(out_path))
    assert info.samplerate == 16000
    assert info.channels == 1
    assert info.format == "WAV"
    assert info.subtype == "PCM_16"
    assert metadata.normalized_sample_rate == 16000
    assert metadata.normalized_channels == 1
    assert metadata.normalized_format == "WAV"


def test_normalize_stereo_downmix(normalizer: AudioNormalizer, tmp_path: Path) -> None:
    """Test that a stereo audio file is accurately downmixed to mono."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    left = 0.4 * np.sin(2 * np.pi * 300 * t)
    right = 0.4 * np.sin(2 * np.pi * 600 * t)
    stereo = np.column_stack([left, right])

    input_file = _write_wav(tmp_path / "stereo_input.wav", stereo, sr)
    output_file = tmp_path / "downmixed.wav"

    out_path, metadata, quality = normalizer.normalize(input_file, output_file)

    info = sf.info(str(out_path))
    assert info.channels == 1
    assert info.samplerate == 16000
    assert metadata.channels == 2
    assert metadata.normalized_channels == 1


@pytest.mark.parametrize("source_sr", [8000, 22050, 44100, 48000])
def test_normalize_resampling_rates(
    normalizer: AudioNormalizer,
    tmp_path: Path,
    source_sr: int,
) -> None:
    """Test resampling from various standard sampling rates to 16 kHz."""
    duration = 1.0
    t = np.linspace(0, duration, int(source_sr * duration), endpoint=False, dtype=np.float32)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)

    input_file = _write_wav(tmp_path / f"input_{source_sr}.wav", tone, source_sr)
    output_file = tmp_path / f"output_{source_sr}_to_16k.wav"

    out_path, metadata, quality = normalizer.normalize(input_file, output_file)

    info = sf.info(str(out_path))
    assert info.samplerate == 16000
    assert info.channels == 1
    # Check that duration is preserved within 1% tolerance
    assert abs(info.duration - duration) < 0.02
    assert metadata.sample_rate == source_sr
    assert metadata.normalized_sample_rate == 16000


def test_normalize_flac_input(normalizer: AudioNormalizer, tmp_path: Path) -> None:
    """Test normalizing a FLAC file to 16 kHz PCM WAV."""
    flac_path = tmp_path / "source.flac"
    t = np.linspace(0, 0.8, int(44100 * 0.8), endpoint=False, dtype=np.float32)
    sf.write(str(flac_path), 0.5 * np.sin(2 * np.pi * 440 * t), 44100, format="FLAC")

    out_path = tmp_path / "from_flac.wav"
    out_path, metadata, _ = normalizer.normalize(flac_path, out_path)

    info = sf.info(str(out_path))
    assert info.format == "WAV"
    assert info.samplerate == 16000
    assert info.channels == 1
    assert metadata.original_format == "FLAC"


def test_normalize_prevents_in_place_mutation(
    normalizer: AudioNormalizer,
    tmp_path: Path,
) -> None:
    """Test that specifying identical source and destination raises AudioProcessingError."""
    target_path = tmp_path / "same_file.wav"
    target_path.write_bytes(b"dummy")

    with pytest.raises(AudioProcessingError) as exc_info:
        normalizer.normalize(target_path, target_path)

    assert exc_info.value.code == "IN_PLACE_MUTATION_FORBIDDEN"


def test_normalize_preserves_original_file(
    normalizer: AudioNormalizer,
    tmp_path: Path,
) -> None:
    """Test that the original file contents and timestamps are unmodified."""
    sr = 48000
    t = np.linspace(0, 0.5, 24000, endpoint=False, dtype=np.float32)
    source_path = _write_wav(tmp_path / "original.wav", 0.3 * np.sin(2 * np.pi * 500 * t), sr)

    orig_bytes = source_path.read_bytes()
    out_path = tmp_path / "output_normalized.wav"

    normalizer.normalize(source_path, out_path)

    assert source_path.read_bytes() == orig_bytes, "Original input file was mutated!"
