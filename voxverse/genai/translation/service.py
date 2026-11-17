"""High-level translation service coordinating text translation and ASR integration.

Owner: P2 — Speech Understanding Engineer (Translation & NLP)
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]

from voxverse.config.logging import get_logger
from voxverse.genai.translation.base import BaseTranslationEngine
from voxverse.genai.translation.exceptions import EmptyInputError
from voxverse.genai.translation.indictrans2_engine import IndicTrans2Engine
from voxverse.genai.translation.schemas import (
    TranslationResult,
    TranslationSegment,
)
from voxverse.genai.translation.text_normalizer import IndicTransTextNormalizer

if TYPE_CHECKING:
    from voxverse.genai.asr.schemas import TranscriptionResult

logger = get_logger("translation.service")


class TranslationService:
    """Service orchestrating English-to-Telugu translation and ASR-to-MT handoff.

    Adheres strictly to dependency injection: accepts any BaseTranslationEngine.
    """

    def __init__(self, engine: BaseTranslationEngine | None = None) -> None:
        self.engine = engine or IndicTrans2Engine()
        self.normalizer = IndicTransTextNormalizer()

    def _get_process_memory_mb(self) -> float | None:
        """Measure current process resident memory in megabytes."""
        if psutil is None:
            return None
        try:
            return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)
        except Exception:
            return None

    def translate_text(
        self,
        text: str,
        *,
        source_language: str = "eng_Latn",
        target_language: str = "tel_Telu",
        beam_size: int = 4,
    ) -> TranslationResult:
        """Translate arbitrary English prose into Telugu with sentence-aware chunking.

        Parameters
        ----------
        text:
            English source text.
        source_language:
            FLORES language tag for source (default: 'eng_Latn').
        target_language:
            FLORES language tag for target (default: 'tel_Telu').
        beam_size:
            Beam search width.

        Returns
        -------
        TranslationResult
            Structured result containing translated Telugu text, segments, and performance telemetry.

        Raises
        ------
        EmptyInputError
            If input text is empty or purely whitespace.
        UnsupportedLanguageError
            If language tags are not supported.
        InferenceError
            If model execution fails.
        """
        cleaned_text = self.normalizer.clean_whitespace(text)
        if not cleaned_text:
            raise EmptyInputError()

        logger.info(
            "TranslationService: Translating %d characters (%s -> %s)",
            len(cleaned_text),
            source_language,
            target_language,
        )

        start_time = time.perf_counter()

        # 1. Split into sentence units to avoid truncation on long paragraphs
        sentences = self.normalizer.split_into_sentences(cleaned_text)
        if not sentences:
            sentences = [cleaned_text]

        # 2. Batch translate sentences
        inf_start = time.perf_counter()
        translated_sentences = self.engine.translate_batch(
            sentences,
            source_language=source_language,
            target_language=target_language,
            beam_size=beam_size,
        )
        inf_duration = time.perf_counter() - inf_start

        # 3. Assemble segments and full translated text
        segments: list[TranslationSegment] = []
        for idx, (src_s, tgt_s) in enumerate(zip(sentences, translated_sentences)):
            segments.append(
                TranslationSegment(
                    id=idx,
                    source_text=src_s,
                    target_text=tgt_s,
                    start=None,
                    end=None,
                )
            )

        full_translation = " ".join(translated_sentences).strip()
        total_duration = time.perf_counter() - start_time
        cps = round(len(cleaned_text) / max(0.001, total_duration), 2)

        peak_mem = self._get_process_memory_mb()

        logger.info(
            "Translation complete: %d chars translated in %.2fs (%.1f chars/s)",
            len(cleaned_text),
            total_duration,
            cps,
        )

        return TranslationResult(
            source_text=cleaned_text,
            translated_text=full_translation,
            source_language=source_language,
            target_language=target_language,
            duration_seconds=round(total_duration, 4),
            inference_duration_seconds=round(inf_duration, 4),
            characters_per_second=cps,
            model_info=self.engine.get_model_info(),
            segments=segments,
            peak_memory_mb=peak_mem,
        )

    def translate_transcription(
        self,
        transcription_result: TranscriptionResult,
        *,
        target_language: str = "tel_Telu",
        beam_size: int = 4,
    ) -> TranslationResult:
        """Translate the output of ASRService while preserving timestamp alignments.

        Accepts the TranscriptionResult produced by P2 ASR, maps each timestamped segment
        to a translated Telugu segment, and constructs the complete translated output.

        Parameters
        ----------
        transcription_result:
            Output from ASRService.transcribe_file.
        target_language:
            Target language code (default 'tel_Telu').
        beam_size:
            Beam search width.

        Returns
        -------
        TranslationResult
            Combined translation preserving segment timestamps from ASR.
        """
        raw_text = self.normalizer.clean_whitespace(transcription_result.text)
        if not raw_text:
            raise EmptyInputError("ASR transcription text is empty.")

        logger.info(
            "TranslationService: Translating ASR transcription (%d segments, %d chars)",
            len(transcription_result.segments),
            len(raw_text),
        )

        start_time = time.perf_counter()

        # If ASR segments exist, translate per segment to preserve audio timestamps
        if transcription_result.segments:
            segment_texts = [
                self.normalizer.clean_whitespace(seg.text)
                for seg in transcription_result.segments
            ]
            inf_start = time.perf_counter()
            translated_segment_texts = self.engine.translate_batch(
                segment_texts,
                source_language="eng_Latn",
                target_language=target_language,
                beam_size=beam_size,
            )
            inf_duration = time.perf_counter() - inf_start

            segments: list[TranslationSegment] = []
            for idx, (asr_seg, trans_text) in enumerate(
                zip(transcription_result.segments, translated_segment_texts)
            ):
                segments.append(
                    TranslationSegment(
                        id=idx,
                        source_text=asr_seg.text,
                        target_text=trans_text,
                        start=asr_seg.start,
                        end=asr_seg.end,
                    )
                )

            full_translation = " ".join(translated_segment_texts).strip()

        else:
            # Fallback if no segments in ASR result
            res = self.translate_text(
                raw_text,
                source_language="eng_Latn",
                target_language=target_language,
                beam_size=beam_size,
            )
            return res

        total_duration = time.perf_counter() - start_time
        cps = round(len(raw_text) / max(0.001, total_duration), 2)
        peak_mem = self._get_process_memory_mb()

        return TranslationResult(
            source_text=raw_text,
            translated_text=full_translation,
            source_language="eng_Latn",
            target_language=target_language,
            duration_seconds=round(total_duration, 4),
            inference_duration_seconds=round(inf_duration, 4),
            characters_per_second=cps,
            model_info=self.engine.get_model_info(),
            segments=segments,
            peak_memory_mb=peak_mem,
        )
