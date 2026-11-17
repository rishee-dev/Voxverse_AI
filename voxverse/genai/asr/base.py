"""Abstract base interface for speech recognition engines.

Owner: P2 — Speech Understanding Engineer
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import soundfile as sf

from voxverse.config.logging import get_logger
from voxverse.genai.asr.exceptions import AudioInputError
from voxverse.genai.asr.schemas import (
    ASRModelInfo,
    LanguageValidationResult,
    TranscriptionResult,
)

logger = get_logger("asr.base")

CANONICAL_SAMPLE_RATE = 16000


class BaseASREngine(ABC):
    """Abstract base class establishing the contract for all VoxVerse ASR engines."""

    def __init__(
        self,
        model_name: str,
        download_root: Path | str | None = None,
        cpu_threads: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.download_root = Path(download_root) if download_root else None
        # Default to half of logical cores or minimum 2 for conservative CPU budgeting
        self.cpu_threads = cpu_threads or max(1, min(4, (os.cpu_count() or 4) // 2))

    @abstractmethod
    def load(self) -> None:
        """Load model weights and initialize inference session."""

    @abstractmethod
    def unload(self) -> None:
        """Release model weights and free allocated memory."""

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if model is currently loaded in memory."""

    @abstractmethod
    def get_model_info(self) -> ASRModelInfo:
        """Return technical metadata about the engine and model."""

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path | str,
        *,
        language: str | None = "en",
        beam_size: int = 5,
        temperature: float = 0.0,
    ) -> TranscriptionResult:
        """Transcribe an audio file and return standardized transcription result.

        Parameters
        ----------
        audio_path:
            Path to the input audio file (normalized 16 kHz mono WAV from P1).
        language:
            Language code (e.g., 'en'). None to allow auto-detection if supported.
        beam_size:
            Beam search size for decoding.
        temperature:
            Decoding temperature.
        """

    @abstractmethod
    def detect_language(self, audio_path: Path | str) -> LanguageValidationResult:
        """Detect language and return validation assessment for English speech."""

    def validate_and_load_audio(
        self,
        audio_path: Path | str,
        *,
        enforce_canonical_sr: bool = False,
    ) -> tuple[Path, np.ndarray, float, int]:
        """Verify audio file, decode into float32 numpy array, and validate attributes.

        Returns
        -------
        tuple[Path, np.ndarray, float, int]
            (resolved_path, audio_array, duration_seconds, sample_rate)

        Raises
        ------
        AudioInputError
            If file is missing, empty, corrupt, or unreadable.
        """
        path = Path(audio_path).resolve()
        if not path.exists():
            raise AudioInputError(
                f"Audio file not found: {path}",
                code="AUDIO_FILE_NOT_FOUND",
            )

        if not path.is_file():
            raise AudioInputError(
                f"Audio path is not a file: {path}",
                code="INVALID_AUDIO_PATH",
            )

        file_size = path.stat().st_size
        if file_size == 0:
            raise AudioInputError(
                f"Audio file is empty (0 bytes): {path}",
                code="EMPTY_AUDIO_FILE",
            )

        try:
            audio_data, sr = sf.read(str(path), dtype="float32")
        except Exception as e:
            raise AudioInputError(
                f"Failed to decode audio file {path}: {e}",
                code="UNREADABLE_AUDIO_FILE",
            ) from e

        # Ensure 1D mono
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1, dtype=np.float32)

        duration = float(len(audio_data)) / float(sr) if sr > 0 else 0.0
        if duration <= 0.0 or len(audio_data) == 0:
            raise AudioInputError(
                f"Audio file has zero or negative duration: {duration}s",
                code="ZERO_DURATION_AUDIO",
            )

        if enforce_canonical_sr and sr != CANONICAL_SAMPLE_RATE:
            raise AudioInputError(
                f"Audio sample rate {sr} Hz does not match canonical {CANONICAL_SAMPLE_RATE} Hz. "
                "Please process input through P1 AudioNormalizer first.",
                code="INVALID_SAMPLE_RATE",
            )

        return path, audio_data, duration, sr
