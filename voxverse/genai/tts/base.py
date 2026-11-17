"""Abstract base class interface for Text-to-Speech (TTS) engines.

Owner: P3 — Voice Generation Engineer
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from voxverse.genai.tts.schemas import (
    SpeakerConditioningData,
    TTSModelInfo,
    TTSResult,
)


class BaseTTSEngine(ABC):
    """Contract that all speech synthesis engines in VoxVerse must implement."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device

    @abstractmethod
    def load(self) -> None:
        """Load synthesis model weights, flow-matching DiT, and vocoder into memory."""

    @abstractmethod
    def unload(self) -> None:
        """Unload model from memory and free CPU/RAM resources."""

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check whether the synthesis model is currently resident in memory."""

    @abstractmethod
    def get_model_info(self) -> TTSModelInfo:
        """Return technical metadata about the synthesis engine and loaded model."""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        *,
        conditioning_data: SpeakerConditioningData | None = None,
        nfe_steps: int = 32,
        speed: float = 1.0,
        output_path: str | None = None,
    ) -> TTSResult:
        """Synthesize Telugu speech audio from text, conditioned on an authorized speaker reference.

        Parameters
        ----------
        text:
            Target text to synthesize in Telugu script.
        conditioning_data:
            Optional authorized speaker reference data (waveform path, verbatim transcript,
            consent hash) for zero-shot voice cloning. If None, default/unconditioned voice is used.
        nfe_steps:
            Number of flow-matching function evaluation steps (ODE solver steps).
        speed:
            Speaking rate multiplier (1.0 = normal rate).
        output_path:
            Optional path to save generated output WAV file. If None, a temporary or
            managed path is generated.

        Returns
        -------
        TTSResult
            Synthesis outcome containing audio metadata, duration, RTF, and performance metrics.
        """
