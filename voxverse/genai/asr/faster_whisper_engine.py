"""Faster-Whisper ASR implementation utilizing CTranslate2 on CPU.

Owner: P2 — Speech Understanding Engineer
"""

from __future__ import annotations

import gc
import time
import tracemalloc
from pathlib import Path

import numpy as np

from voxverse.config.logging import get_logger
from voxverse.genai.asr.base import BaseASREngine
from voxverse.genai.asr.exceptions import (
    IncompatibleComputeTypeError,
    ModelLoadError,
    TranscriptionError,
)
from voxverse.genai.asr.schemas import (
    ASRModelInfo,
    LanguageValidationResult,
    TranscriptionResult,
    TranscriptionSegment,
)

logger = get_logger("asr.faster_whisper")

DEFAULT_DOWNLOAD_ROOT = Path("models/faster-whisper")
VALID_CPU_COMPUTE_TYPES = {"int8", "float32", "int8_float32"}


class FasterWhisperEngine(BaseASREngine):
    """ASR Engine powered by faster-whisper and CTranslate2 with INT8 CPU optimization."""

    def __init__(
        self,
        model_name: str = "tiny.en",
        compute_type: str = "int8",
        download_root: Path | str | None = None,
        cpu_threads: int | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            download_root=download_root or DEFAULT_DOWNLOAD_ROOT,
            cpu_threads=cpu_threads,
        )
        if compute_type not in VALID_CPU_COMPUTE_TYPES:
            raise IncompatibleComputeTypeError(
                f"Compute type '{compute_type}' is not supported on CPU. Valid options: {VALID_CPU_COMPUTE_TYPES}",
                code="UNSUPPORTED_CPU_COMPUTE_TYPE",
            )
        self.compute_type = compute_type
        self.is_multilingual = not model_name.endswith(".en")
        self._model = None

    def load(self) -> None:
        """Initialize and load the CTranslate2 model."""
        if self._model is not None:
            return

        logger.info(
            "Loading faster-whisper model '%s' (compute_type=%s, threads=%d)...",
            self.model_name,
            self.compute_type,
            self.cpu_threads,
        )
        try:
            from faster_whisper import WhisperModel

            root = str(self.download_root) if self.download_root else None
            self._model = WhisperModel(
                self.model_name,
                device="cpu",
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
                download_root=root,
            )
            logger.info("Successfully loaded faster-whisper model '%s'", self.model_name)
        except Exception as e:
            logger.error("Failed to load faster-whisper model '%s': %s", self.model_name, e)
            raise ModelLoadError(
                f"Failed to load faster-whisper model '{self.model_name}': {e}",
                code="FASTER_WHISPER_LOAD_FAILED",
            ) from e

    def unload(self) -> None:
        """Release CTranslate2 resources and free memory."""
        if self._model is not None:
            logger.info("Unloading faster-whisper model '%s'", self.model_name)
            del self._model
            self._model = None
            gc.collect()

    def is_loaded(self) -> bool:
        """Return True if model is loaded and ready for inference."""
        return self._model is not None

    def get_model_info(self) -> ASRModelInfo:
        """Return technical metadata about this faster-whisper instance."""
        return ASRModelInfo(
            model_name=self.model_name,
            backend="faster-whisper",
            compute_type=self.compute_type,
            is_multilingual=self.is_multilingual,
            device="cpu",
            cpu_threads=self.cpu_threads,
        )

    def detect_language(self, audio_path: Path | str) -> LanguageValidationResult:
        """Detect language using the underlying model if multilingual."""
        path, audio_data, duration, sr = self.validate_and_load_audio(audio_path)

        if not self.is_multilingual:
            # English-only (.en) model variants have only English tokens and cannot classify language
            return LanguageValidationResult(
                is_english=True,
                detected_language="en",
                language_probability=1.0,
                status="ASSUMED_ENGLISH",
                reason=(
                    f"Model '{self.model_name}' is an English-only model variant (.en) "
                    "with no multilingual language classification head; English input is assumed."
                ),
            )

        if not self.is_loaded():
            self.load()

        try:
            lang, prob, _ = self._model.detect_language(audio_data)
            prob_float = float(prob)

            if lang == "en":
                status = "VERIFIED_ENGLISH" if prob_float >= 0.70 else "UNCERTAIN"
                return LanguageValidationResult(
                    is_english=True,
                    detected_language="en",
                    language_probability=prob_float,
                    status=status,
                    reason=f"Detected English with probability {prob_float:.3f}.",
                )
            else:
                status = "REJECTED_NON_ENGLISH" if prob_float >= 0.50 else "UNCERTAIN"
                return LanguageValidationResult(
                    is_english=False,
                    detected_language=str(lang),
                    language_probability=prob_float,
                    status=status,
                    reason=f"Detected language '{lang}' with probability {prob_float:.3f} (threshold: 0.50).",
                )
        except Exception as e:
            logger.error("Language detection failed for %s: %s", path, e)
            raise TranscriptionError(
                f"Language detection failed on {path}: {e}",
                code="LANGUAGE_DETECTION_FAILED",
            ) from e

    def transcribe(
        self,
        audio_path: Path | str,
        *,
        language: str | None = "en",
        beam_size: int = 5,
        temperature: float = 0.0,
    ) -> TranscriptionResult:
        """Transcribe audio using faster-whisper on CPU."""
        path, audio_data, audio_duration, sr = self.validate_and_load_audio(audio_path)

        if not self.is_loaded():
            self.load()

        target_lang = "en" if not self.is_multilingual else language

        tracemalloc.start()
        start_time = time.perf_counter()

        try:
            # Pass float32 numpy array to bypass PyAV / FFmpeg dependencies
            segments_gen, info = self._model.transcribe(
                audio_data,
                language=target_lang,
                beam_size=beam_size,
                temperature=temperature,
            )

            segments: list[TranscriptionSegment] = []
            transcription_texts: list[str] = []

            for seg in segments_gen:
                seg_text = seg.text.strip()
                if seg_text:
                    transcription_texts.append(seg_text)
                segments.append(
                    TranscriptionSegment(
                        id=seg.id,
                        seek=seg.seek,
                        start=round(float(seg.start), 3),
                        end=round(float(seg.end), 3),
                        text=seg_text,
                        avg_logprob=round(float(seg.avg_logprob), 4) if seg.avg_logprob is not None else None,
                        no_speech_prob=round(float(seg.no_speech_prob), 4) if seg.no_speech_prob is not None else None,
                        compression_ratio=round(float(seg.compression_ratio), 4) if seg.compression_ratio is not None else None,
                    )
                )

            inference_duration = time.perf_counter() - start_time
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            full_text = " ".join(transcription_texts).strip()
            rtf = inference_duration / audio_duration if audio_duration > 0 else 0.0
            peak_mb = peak_bytes / (1024 * 1024)

            lang_code = info.language if hasattr(info, "language") and info.language else (target_lang or "en")
            lang_prob = float(info.language_probability) if hasattr(info, "language_probability") and info.language_probability is not None else None

            lang_val = None
            if self.is_multilingual:
                is_eng = (lang_code == "en")
                val_status = "VERIFIED_ENGLISH" if (is_eng and (lang_prob or 0.0) >= 0.70) else ("REJECTED_NON_ENGLISH" if not is_eng else "UNCERTAIN")
                lang_val = LanguageValidationResult(
                    is_english=is_eng,
                    detected_language=lang_code,
                    language_probability=lang_prob,
                    status=val_status,
                )

            return TranscriptionResult(
                text=full_text,
                language=lang_code,
                language_probability=lang_prob,
                duration_seconds=round(audio_duration, 3),
                inference_duration_seconds=round(inference_duration, 4),
                real_time_factor=round(rtf, 4),
                model_info=self.get_model_info(),
                segments=segments,
                language_validation=lang_val,
                peak_memory_mb=round(peak_mb, 2),
            )

        except Exception as e:
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            logger.error("Faster-whisper transcription failed for %s: %s", path, e)
            raise TranscriptionError(
                f"Faster-whisper transcription failed for '{path.name}': {e}",
                code="FASTER_WHISPER_TRANSCRIPTION_FAILED",
            ) from e
