"""IndicTrans2 translation engine implementation using Rotary IndicTrans.

Owner: P2 — Speech Understanding Engineer (Translation & NLP)
"""

from __future__ import annotations

import gc
import os
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from voxverse.config.logging import get_logger
from voxverse.genai.translation.base import BaseTranslationEngine
from voxverse.genai.translation.exceptions import (
    InferenceError,
    ModelLoadError,
    UnsupportedLanguageError,
)
from voxverse.genai.translation.schemas import (
    SUPPORTED_SOURCE_LANGUAGES,
    SUPPORTED_TARGET_LANGUAGES,
    TranslationModelInfo,
)
from voxverse.genai.translation.text_normalizer import IndicTransTextNormalizer

logger = get_logger("translation.indictrans2")

DEFAULT_MODEL_NAME = "prajdabre/rotary-indictrans2-en-indic-dist-200M"


class IndicTrans2Engine(BaseTranslationEngine):
    """Production implementation of English-to-Indic translation using IndicTrans2 200M.

    Uses Rotary IndicTrans2 architecture running on CPU float32 with beam search decoding.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        num_threads: int | None = None,
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self.num_threads = num_threads or max(1, min(4, (os.cpu_count() or 4) // 2))
        self.tokenizer: Any = None
        self.model: Any = None
        self.normalizer = IndicTransTextNormalizer()
        self._parameter_count: int = 0

    def is_loaded(self) -> bool:
        """Return True if model and tokenizer are initialized in memory."""
        return self.model is not None and self.tokenizer is not None

    def load(self) -> None:
        """Load tokenizer and model weights into memory on CPU."""
        if self.is_loaded():
            logger.debug("IndicTrans2Engine already loaded.")
            return

        logger.info(
            "Loading IndicTrans2 model from '%s' on %s (threads=%d)...",
            self.model_name,
            self.device,
            self.num_threads,
        )

        try:
            torch.set_num_threads(self.num_threads)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                dtype=torch.float32,
            )
            self.model.eval()
            self._parameter_count = sum(p.numel() for p in self.model.parameters())
            logger.info(
                "IndicTrans2 loaded successfully: %d parameters.",
                self._parameter_count,
            )
        except Exception as e:
            logger.error("Failed to load IndicTrans2 model: %s", e, exc_info=True)
            self.unload()
            raise ModelLoadError(f"Failed to load translation model '{self.model_name}': {e}") from e

    def unload(self) -> None:
        """Unload model and tokenizer from memory and trigger garbage collection."""
        if not self.is_loaded():
            return

        logger.info("Unloading IndicTrans2 model from memory...")
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("IndicTrans2 unloaded.")

    def get_model_info(self) -> TranslationModelInfo:
        """Return technical metadata about the engine."""
        return TranslationModelInfo(
            model_name=self.model_name,
            backend="transformers-rotary-indictrans2",
            device=self.device,
            compute_type="float32",
            parameter_count=self._parameter_count or 211776512,
            max_sequence_length=256,
        )

    def _validate_languages(self, source_language: str, target_language: str) -> None:
        """Verify that the requested source and target language codes are valid."""
        if source_language not in SUPPORTED_SOURCE_LANGUAGES:
            raise UnsupportedLanguageError(source_language, sorted(SUPPORTED_SOURCE_LANGUAGES))
        if target_language not in SUPPORTED_TARGET_LANGUAGES:
            raise UnsupportedLanguageError(target_language, sorted(SUPPORTED_TARGET_LANGUAGES))

    def translate(
        self,
        text: str,
        *,
        source_language: str = "eng_Latn",
        target_language: str = "tel_Telu",
        beam_size: int = 4,
        max_length: int = 256,
    ) -> str:
        """Translate a single English string into Telugu."""
        results = self.translate_batch(
            [text],
            source_language=source_language,
            target_language=target_language,
            beam_size=beam_size,
            max_length=max_length,
        )
        return results[0] if results else ""

    def translate_batch(
        self,
        texts: list[str],
        *,
        source_language: str = "eng_Latn",
        target_language: str = "tel_Telu",
        beam_size: int = 4,
        max_length: int = 256,
    ) -> list[str]:
        """Translate a batch of English sentences into Telugu."""
        self._validate_languages(source_language, target_language)

        if not texts:
            return []

        if not self.is_loaded():
            self.load()

        formatted_inputs: list[str] = []
        placeholder_maps: list[dict[str, str]] = []

        # 1. Preprocess each sentence with entity protection and language tags
        for t in texts:
            formatted, p_map = self.normalizer.preprocess_sentence(
                t,
                source_language=source_language,
                target_language=target_language,
            )
            formatted_inputs.append(formatted)
            placeholder_maps.append(p_map)

        try:
            # 2. Tokenize batch
            inputs = self.tokenizer(
                formatted_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )

            # 3. Model generation on CPU
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=beam_size,
                    use_cache=False,
                )

            # 4. Token decode
            decoded_batch = self.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
            )

            # 5. Postprocess (Devanagari -> Telugu script + placeholder restoration)
            translated_sentences: list[str] = []
            for raw_decoded, p_map in zip(decoded_batch, placeholder_maps):
                processed = self.normalizer.postprocess_sentence(
                    raw_decoded,
                    target_language=target_language,
                    placeholder_map=p_map,
                )
                translated_sentences.append(processed)

            return translated_sentences

        except Exception as e:
            logger.error("Inference failed during translate_batch: %s", e, exc_info=True)
            raise InferenceError(f"Translation inference failed: {e}") from e
