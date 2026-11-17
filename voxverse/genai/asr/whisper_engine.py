"""OpenAI Whisper ASR implementation running PyTorch on CPU.

Owner: P2 — Speech Understanding Engineer
"""

from __future__ import annotations

import gc
import time
import tracemalloc
from pathlib import Path

import numpy as np
import torch

from voxverse.config.logging import get_logger
from voxverse.genai.asr.base import BaseASREngine
from voxverse.genai.asr.exceptions import (
    ModelLoadError,
    TranscriptionError,
)
from voxverse.genai.asr.schemas import (
    ASRModelInfo,
    LanguageValidationResult,
    TranscriptionResult,
    TranscriptionSegment,
)

logger = get_logger("asr.openai_whisper")

DEFAULT_DOWNLOAD_ROOT = Path("models/whisper")


class OpenAIWhisperEngine(BaseASREngine):
    """ASR Engine powered by the official open-source openai-whisper package on CPU."""

    def __init__(
        self,
        model_name: str = "tiny.en",
        download_root: Path | str | None = None,
        cpu_threads: int | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            download_root=download_root or DEFAULT_DOWNLOAD_ROOT,
            cpu_threads=cpu_threads,
        )
        self.is_multilingual = not model_name.endswith(".en")
        self._model = None

    def load(self) -> None:
        """Load PyTorch Whisper model weights."""
        if self._model is not None:
            return

        logger.info(
            "Loading OpenAI Whisper model '%s' on CPU (threads=%d)...",
            self.model_name,
            self.cpu_threads,
        )
        try:
            import whisper

            torch.set_num_threads(self.cpu_threads)
            root = str(self.download_root) if self.download_root else None
            self._model = whisper.load_model(
                self.model_name,
                device="cpu",
                download_root=root,
            )
            logger.info("Successfully loaded OpenAI Whisper model '%s'", self.model_name)
        except Exception as e:
            logger.error("Failed to load OpenAI Whisper model '%s': %s", self.model_name, e)
            raise ModelLoadError(
                f"Failed to load OpenAI Whisper model '{self.model_name}': {e}",
                code="OPENAI_WHISPER_LOAD_FAILED",
            ) from e

    def unload(self) -> None:
        """Release PyTorch model weights and trigger garbage collection."""
        if self._model is not None:
            logger.info("Unloading OpenAI Whisper model '%s'", self.model_name)
            del self._model
            self._model = None
            gc.collect()

    def is_loaded(self) -> bool:
        """Return True if model is loaded and ready for inference."""
        return self._model is not None

    def get_model_info(self) -> ASRModelInfo:
        """Return technical metadata about this OpenAI Whisper instance."""
        return ASRModelInfo(
            model_name=self.model_name,
            backend="openai-whisper",
            compute_type="float32",
            is_multilingual=self.is_multilingual,
            device="cpu",
            cpu_threads=self.cpu_threads,
        )

    def detect_language(self, audio_path: Path | str) -> LanguageValidationResult:
        """Detect language if model is multilingual."""
        path, audio_data, duration, sr = self.validate_and_load_audio(audio_path)

        if not self.is_multilingual:
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
            import whisper

            audio_trimmed = whisper.pad_or_trim(audio_data)
            mel = whisper.log_mel_spectrogram(audio_trimmed).to(self._model.device)
            _, probs = self._model.detect_language(mel)

            detected_lang = max(probs, key=probs.get)
            prob_float = float(probs[detected_lang])

            if detected_lang == "en":
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
                    detected_language=detected_lang,
                    language_probability=prob_float,
                    status=status,
                    reason=f"Detected language '{detected_lang}' with probability {prob_float:.3f} (threshold: 0.50).",
                )

        except Exception as e:
            logger.error("OpenAI Whisper language detection failed for %s: %s", path, e)
            raise TranscriptionError(
                f"OpenAI Whisper language detection failed on {path}: {e}",
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
        """Transcribe audio using official OpenAI Whisper on CPU."""
        path, audio_data, audio_duration, sr = self.validate_and_load_audio(audio_path)

        if not self.is_loaded():
            self.load()

        target_lang = "en" if not self.is_multilingual else language

        tracemalloc.start()
        start_time = time.perf_counter()

        try:
            # Pass float32 numpy array to bypass external ffmpeg dependency
            result = self._model.transcribe(
                audio_data,
                language=target_lang,
                beam_size=beam_size,
                temperature=temperature,
                fp16=False,
            )

            inference_duration = time.perf_counter() - start_time
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            raw_segments = result.get("segments", [])
            segments: list[TranscriptionSegment] = []

            for seg in raw_segments:
                seg_text = seg.get("text", "").strip()
                segments.append(
                    TranscriptionSegment(
                        id=int(seg.get("id", len(segments))),
                        seek=int(seg.get("seek", 0)),
                        start=round(float(seg.get("start", 0.0)), 3),
                        end=round(float(seg.get("end", 0.0)), 3),
                        text=seg_text,
                        avg_logprob=round(float(seg.get("avg_logprob")), 4) if seg.get("avg_logprob") is not None else None,
                        no_speech_prob=round(float(seg.get("no_speech_prob")), 4) if seg.get("no_speech_prob") is not None else None,
                        compression_ratio=round(float(seg.get("compression_ratio")), 4) if seg.get("compression_ratio") is not None else None,
                    )
                )

            full_text = result.get("text", "").strip()
            rtf = inference_duration / audio_duration if audio_duration > 0 else 0.0
            peak_mb = peak_bytes / (1024 * 1024)

            lang_code = result.get("language", target_lang or "en")

            return TranscriptionResult(
                text=full_text,
                language=lang_code,
                language_probability=None,
                duration_seconds=round(audio_duration, 3),
                inference_duration_seconds=round(inference_duration, 4),
                real_time_factor=round(rtf, 4),
                model_info=self.get_model_info(),
                segments=segments,
                language_validation=None,
                peak_memory_mb=round(peak_mb, 2),
            )

        except Exception as e:
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            logger.error("OpenAI Whisper transcription failed for %s: %s", path, e)
            raise TranscriptionError(
                f"OpenAI Whisper transcription failed for '{path.name}': {e}",
                code="OPENAI_WHISPER_TRANSCRIPTION_FAILED",
            ) from e
