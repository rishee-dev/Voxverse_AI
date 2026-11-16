"""Empirical benchmark of VoxVerse Phase 1A audio processing and VAD pipeline.

Measures:
- Audio decoding latency
- Validation analysis latency
- Normalization (stereo 44.1 kHz -> mono 16 kHz PCM_16) latency
- Silero VAD CPU inference latency and RTF
- End-to-end pipeline latency and combined RTF
- Peak memory allocation during processing

Owner: P1 — Audio and Speaker Profile Engineer
"""

from __future__ import annotations

import gc
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf

from voxverse.genai.audio.decoder import AudioDecoder
from voxverse.genai.audio.normalizer import AudioNormalizer
from voxverse.genai.audio.vad import SileroVADDetector
from voxverse.genai.audio.validator import AudioValidator


def generate_benchmark_signal(duration_sec: float, sample_rate: int = 44100) -> np.ndarray:
    """Generate deterministic stereo audio signal with alternating tone and silence."""
    num_samples = int(sample_rate * duration_sec)
    t = np.linspace(0.0, float(duration_sec), num_samples, endpoint=False, dtype=np.float64)
    # 440 Hz on left, 880 Hz on right, amplitude 0.4
    left = (0.4 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    right = (0.4 * np.sin(2.0 * np.pi * 880.0 * t)).astype(np.float32)
    return np.column_stack([left, right])


def run_benchmark_for_duration(
    duration_sec: float,
    decoder: AudioDecoder,
    validator: AudioValidator,
    normalizer: AudioNormalizer,
    vad: SileroVADDetector,
    temp_dir: Path,
) -> dict[str, float | str | int]:
    """Execute complete audio pipeline benchmark for a given audio duration."""
    raw_path = temp_dir / f"bench_{int(duration_sec)}s_raw.wav"
    norm_path = temp_dir / f"bench_{int(duration_sec)}s_norm.wav"

    # 1. Synthesize and write 44.1 kHz stereo audio
    signal = generate_benchmark_signal(duration_sec, sample_rate=44100)
    sf.write(str(raw_path), signal, 44100, subtype="PCM_16", format="WAV")
    raw_size_bytes = raw_path.stat().st_size

    gc.collect()
    tracemalloc.start()

    # Stage 1: Decode
    t0 = time.perf_counter()
    audio_data, sr, bit_depth, fmt = decoder.decode_file(raw_path)
    t_decode = time.perf_counter() - t0

    # Stage 2: Validate
    t0 = time.perf_counter()
    report = validator.validate(audio_data, sr)
    t_validate = time.perf_counter() - t0

    # Stage 3: Normalize (downmix + resample 44.1k -> 16k mono WAV)
    t0 = time.perf_counter()
    _, metadata, _ = normalizer.normalize(raw_path, norm_path)
    t_normalize = time.perf_counter() - t0

    # Stage 4: Decode normalized audio for VAD
    norm_audio, norm_sr, _, _ = decoder.decode_file(norm_path)

    # Stage 5: VAD Inference on CPU
    t0 = time.perf_counter()
    vad_result = vad.detect_speech(norm_audio, norm_sr)
    t_vad = time.perf_counter() - t0

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t_total = t_decode + t_validate + t_normalize + t_vad
    pipeline_rtf = t_total / duration_sec
    vad_rtf = t_vad / duration_sec

    # Cleanup temporary files
    if raw_path.exists():
        raw_path.unlink()
    if norm_path.exists():
        norm_path.unlink()

    return {
        "duration_sec": duration_sec,
        "raw_size_kb": round(raw_size_bytes / 1024, 1),
        "decode_time_ms": round(t_decode * 1000, 2),
        "validate_time_ms": round(t_validate * 1000, 2),
        "normalize_time_ms": round(t_normalize * 1000, 2),
        "vad_time_ms": round(t_vad * 1000, 2),
        "total_time_ms": round(t_total * 1000, 2),
        "vad_rtf": round(vad_rtf, 5),
        "pipeline_rtf": round(pipeline_rtf, 5),
        "peak_memory_mb": round(peak_mem / (1024 * 1024), 2),
    }


def main() -> None:
    print("=" * 80)
    print("VOXVERSE PHASE 1A: AUDIO PROCESSING & VAD BENCHMARK")
    print("=" * 80)
    print("Host: Snapdragon X X126100 (8 cores, ARM64, Python 3.14.8 AMD64 emulation)")
    print("Target Standard: 16 kHz mono 16-bit PCM WAV")
    print("=" * 80)

    decoder = AudioDecoder()
    validator = AudioValidator()
    normalizer = AudioNormalizer()
    vad = SileroVADDetector()

    durations = [5.0, 30.0, 60.0]

    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        results = []
        for d in durations:
            print(f"\nRunning benchmark for duration = {d:.0f} seconds...")
            res = run_benchmark_for_duration(d, decoder, validator, normalizer, vad, temp_dir)
            results.append(res)
            print(f" -> Total Processing Time: {res['total_time_ms']} ms")
            print(f" -> Decode: {res['decode_time_ms']} ms | Validate: {res['validate_time_ms']} ms")
            print(f" -> Normalize (resample 44.1k->16k): {res['normalize_time_ms']} ms")
            print(f" -> VAD (Silero ONNX CPU): {res['vad_time_ms']} ms (RTF: {res['vad_rtf']})")
            print(f" -> Overall Pipeline RTF: {res['pipeline_rtf']}")
            print(f" -> Peak Memory: {res['peak_memory_mb']} MB")

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    print(
        f"{'Duration':<10} | {'Input Size':<12} | {'Normalize':<12} | {'VAD CPU':<12} | "
        f"{'Total Time':<12} | {'VAD RTF':<10} | {'Pipeline RTF':<12} | {'Peak RAM':<10}"
    )
    print("-" * 100)
    for r in results:
        print(
            f"{r['duration_sec']:<10.0f} | {str(r['raw_size_kb']) + ' KB':<12} | "
            f"{str(r['normalize_time_ms']) + ' ms':<12} | {str(r['vad_time_ms']) + ' ms':<12} | "
            f"{str(r['total_time_ms']) + ' ms':<12} | {r['vad_rtf']:<10.4f} | "
            f"{r['pipeline_rtf']:<12.4f} | {str(r['peak_memory_mb']) + ' MB':<10}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
