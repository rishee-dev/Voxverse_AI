"""Abstract base class interface for translation engines.

Owner: P2 — Speech Understanding Engineer (Translation & NLP)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from voxverse.genai.translation.schemas import TranslationModelInfo


class BaseTranslationEngine(ABC):
    """Contract that all translation engines in VoxVerse must implement."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device

    @abstractmethod
    def load(self) -> None:
        """Load model weights and tokenizer into memory."""

    @abstractmethod
    def unload(self) -> None:
        """Unload model from memory and free hardware resources."""

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check whether the model is currently resident in memory."""

    @abstractmethod
    def get_model_info(self) -> TranslationModelInfo:
        """Return technical metadata about the engine and loaded model."""

    @abstractmethod
    def translate(
        self,
        text: str,
        *,
        source_language: str = "eng_Latn",
        target_language: str = "tel_Telu",
        beam_size: int = 4,
        max_length: int = 256,
    ) -> str:
        """Translate a single input string.

        Parameters
        ----------
        text:
            Source English text.
        source_language:
            Source language code (default: 'eng_Latn').
        target_language:
            Target language code (default: 'tel_Telu').
        beam_size:
            Beam search size.
        max_length:
            Maximum token generation length.

        Returns
        -------
        str
            Translated Telugu text.
        """

    @abstractmethod
    def translate_batch(
        self,
        texts: list[str],
        *,
        source_language: str = "eng_Latn",
        target_language: str = "tel_Telu",
        beam_size: int = 4,
        max_length: int = 256,
    ) -> list[str]:
        """Translate a batch of text strings simultaneously.

        Parameters
        ----------
        texts:
            List of source strings.
        source_language:
            Source language code.
        target_language:
            Target language code.
        beam_size:
            Beam search width.
        max_length:
            Maximum token length.

        Returns
        -------
        list[str]
            List of translated strings.
        """
