"""Voice Activity Detection (VAD) engine powered by Silero VAD on ONNX Runtime CPU.

Owner: P1 — Audio and Speaker Profile Engineer
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from voxverse.config.logging import get_logger
from voxverse.genai.audio.exceptions import VADInferenceError
from voxverse.genai.audio.schemas import SpeechSegment, VADResult

logger = get_logger("audio.vad")

DEFAULT_MODEL_PATH: Path = Path("models/silero_vad.onnx")
WINDOW_SIZE_SAMPLES: int = 512  # 32 ms at 16,000 Hz
REQUIRED_SAMPLE_RATE: int = 16000
RECURRENT_STATE_SHAPE: tuple[int, int, int] = (2, 1, 128)


class SileroVADDetector:
    """Detects speech regions in 16 kHz mono audio using Silero VAD on ONNX Runtime CPU.

    Note on Synthetic Testing:
    Silero VAD is a deep neural model trained on human vocal tract acoustics.
    Synthetic signals (silence, sine waves, constant tones) are properly classified
    as non-speech. Verification with synthetic signals validates runtime execution,
    stability, latency, and false positive rejection. Real human speech validation
    is required to benchmark true positive speech extraction.
    """

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        threshold: float = 0.5,
        neg_threshold: float = 0.35,
        min_speech_duration_ms: float = 250.0,
        min_silence_duration_ms: float = 100.0,
        speech_pad_ms: float = 30.0,
    ) -> None:
        """Initialize the Silero VAD session.

        Parameters
        ----------
        model_path:
            Filesystem path to the silero_vad.onnx model weight file.
        threshold:
            Probability threshold to trigger the start of a speech segment.
        neg_threshold:
            Probability threshold below which speech is considered ended.
        min_speech_duration_ms:
            Minimum duration in ms for a segment to be kept.
        min_silence_duration_ms:
            Minimum duration in ms of non-speech before closing a segment.
        speech_pad_ms:
            Padding in ms added to the beginning and end of detected speech segments.
        """
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.neg_threshold = neg_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms

        if not self.model_path.exists():
            raise VADInferenceError(
                f"Silero VAD model file not found at '{self.model_path}'.",
                code="MODEL_NOT_FOUND",
            )

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            logger.info("Initialized Silero VAD session on CPUExecutionProvider")
        except Exception as err:
            raise VADInferenceError(
                f"Failed to create ONNX Runtime session for Silero VAD: {err}",
                code="ORT_SESSION_INIT_FAILED",
            ) from err

    def detect_speech(
        self,
        audio_data: np.ndarray,
        sample_rate: int = REQUIRED_SAMPLE_RATE,
    ) -> VADResult:
        """Process 16 kHz mono audio and return timestamped speech segments.

        Parameters
        ----------
        audio_data:
            1D float32 NumPy array of audio samples in range [-1.0, 1.0].
        sample_rate:
            Audio sampling rate in Hertz. Must be 16,000 Hz.

        Returns
        -------
        VADResult
            Detailed results including segment list, speech ratio, and execution timings.

        Raises
        ------
        VADInferenceError
            If audio format is invalid or inference fails.
        """
        if sample_rate != REQUIRED_SAMPLE_RATE:
            raise VADInferenceError(
                f"Silero VAD requires sample_rate={REQUIRED_SAMPLE_RATE} Hz, got {sample_rate} Hz.",
                code="INVALID_SAMPLE_RATE",
            )

        if audio_data is None or audio_data.size == 0:
            return VADResult(
                audio_duration_seconds=0.0,
                speech_duration_seconds=0.0,
                speech_ratio=0.0,
                segments=[],
                processing_time_ms=0.0,
                real_time_factor=0.0,
            )

        # Flatten if 2D
        flat_audio = audio_data.flatten().astype(np.float32)
        total_samples = flat_audio.size
        audio_duration_sec = total_samples / float(REQUIRED_SAMPLE_RATE)

        # Prepare ONNX state and sample rate input
        state = np.zeros(RECURRENT_STATE_SHAPE, dtype=np.float32)
        sr_tensor = np.array(REQUIRED_SAMPLE_RATE, dtype=np.int64)

        t_start = time.perf_counter()

        # Step 1: Compute speech probabilities for each 512-sample (32ms) window
        chunk_probs: list[tuple[float, float]] = []  # (timestamp_ms, prob)
        try:
            for offset in range(0, total_samples, WINDOW_SIZE_SAMPLES):
                chunk = flat_audio[offset : offset + WINDOW_SIZE_SAMPLES]
                if len(chunk) < WINDOW_SIZE_SAMPLES:
                    chunk = np.pad(chunk, (0, WINDOW_SIZE_SAMPLES - len(chunk)))

                chunk = chunk.reshape(1, WINDOW_SIZE_SAMPLES)
                out, state = self.session.run(
                    None,
                    {"input": chunk, "state": state, "sr": sr_tensor},
                )
                prob = float(out[0][0])
                time_ms = (offset / float(REQUIRED_SAMPLE_RATE)) * 1000.0
                chunk_probs.append((time_ms, prob))

        except Exception as err:
            raise VADInferenceError(
                f"Inference execution failed during VAD chunk processing: {err}",
                code="INFERENCE_EXECUTION_FAILURE",
            ) from err

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # Step 2: Extract speech intervals using state machine with hysteresis
        raw_segments: list[SpeechSegment] = []
        triggered = False
        current_speech_start = 0.0
        current_probs: list[float] = []

        window_duration_ms = (WINDOW_SIZE_SAMPLES / float(REQUIRED_SAMPLE_RATE)) * 1000.0

        for time_ms, prob in chunk_probs:
            if not triggered:
                if prob >= self.threshold:
                    triggered = True
                    current_speech_start = time_ms
                    current_probs = [prob]
            else:
                current_probs.append(prob)
                if prob < self.neg_threshold:
                    # Potential end of speech segment
                    speech_end = time_ms + window_duration_ms
                    segment_duration = speech_end - current_speech_start
                    if segment_duration >= self.min_speech_duration_ms:
                        mean_conf = float(np.mean(current_probs)) if current_probs else prob
                        raw_segments.append(
                            SpeechSegment(
                                start_ms=max(0.0, current_speech_start - self.speech_pad_ms),
                                end_ms=min(audio_duration_sec * 1000.0, speech_end + self.speech_pad_ms),
                                confidence=round(mean_conf, 4),
                            )
                        )
                    triggered = False
                    current_probs = []

        # Handle segment running to the very end of audio
        if triggered:
            speech_end = audio_duration_sec * 1000.0
            if (speech_end - current_speech_start) >= self.min_speech_duration_ms:
                mean_conf = float(np.mean(current_probs)) if current_probs else 0.5
                raw_segments.append(
                    SpeechSegment(
                        start_ms=max(0.0, current_speech_start - self.speech_pad_ms),
                        end_ms=speech_end,
                        confidence=round(mean_conf, 4),
                    )
                )

        # Step 3: Merge segments that are closer together than min_silence_duration_ms
        merged_segments: list[SpeechSegment] = []
        for seg in raw_segments:
            if not merged_segments:
                merged_segments.append(seg)
            else:
                prev = merged_segments[-1]
                gap = seg.start_ms - prev.end_ms
                if gap < self.min_silence_duration_ms:
                    # Merge into prev
                    merged_segments[-1] = SpeechSegment(
                        start_ms=prev.start_ms,
                        end_ms=max(prev.end_ms, seg.end_ms),
                        confidence=round((prev.confidence + seg.confidence) / 2.0, 4),
                    )
                else:
                    merged_segments.append(seg)

        # Step 4: Calculate total speech duration and real-time factor
        speech_duration_sec = sum(s.duration_ms for s in merged_segments) / 1000.0
        speech_ratio = speech_duration_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0
        rtf = (elapsed_ms / 1000.0) / audio_duration_sec if audio_duration_sec > 0 else 0.0

        result = VADResult(
            audio_duration_seconds=round(audio_duration_sec, 3),
            speech_duration_seconds=round(speech_duration_sec, 3),
            speech_ratio=round(speech_ratio, 4),
            segments=merged_segments,
            processing_time_ms=round(elapsed_ms, 2),
            real_time_factor=round(rtf, 5),
        )

        logger.info(
            "VAD inference: duration=%.2fs, speech=%.2fs (%.1f%%), segments=%d, latency=%.1fms, RTF=%.4f",
            result.audio_duration_seconds,
            result.speech_duration_seconds,
            result.speech_ratio * 100.0,
            len(result.segments),
            result.processing_time_ms,
            result.real_time_factor,
        )

        return result
