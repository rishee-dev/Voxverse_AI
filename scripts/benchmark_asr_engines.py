"""Empirical benchmarking script for VoxVerse ASR engines on Snapdragon X CPU.

Compares:
1. faster-whisper (tiny.en, int8)
2. faster-whisper (tiny.en, float32)
3. faster-whisper (base.en, int8)
4. openai-whisper (tiny.en, float32)
5. openai-whisper (base.en, float32)

Evaluates on actual speech from LibriSpeech test-clean subset (57.1s total audio).
Outputs full metrics: WER, CER, S, D, I, Latency, RTF, Peak RAM, and Model Load Time.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure voxverse is in python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voxverse.genai.asr.faster_whisper_engine import FasterWhisperEngine
from voxverse.genai.asr.metrics import calculate_cer, calculate_wer, normalize_transcript
from voxverse.genai.asr.whisper_engine import OpenAIWhisperEngine

MANIFEST_PATH = Path("data/benchmark/librispeech/manifest.json")


def run_benchmark():
    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found at {MANIFEST_PATH}. Run prepare_librispeech_subset.py first.")
        sys.exit(1)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    samples = manifest["samples"]
    total_audio_duration = manifest["total_duration_seconds"]
    print("=" * 80)
    print("VOXVERSE ASR EMPIRICAL BENCHMARK — CPU EXECUTION")
    print(f"Dataset: {manifest['dataset_name']} ({len(samples)} samples, {total_audio_duration:.2f}s total audio)")
    print(f"Hardware: Windows 11 Qualcomm Snapdragon X (8 cores, CPU only)")
    print("=" * 80)

    # Candidate Configurations
    configurations = [
        {
            "name": "faster-whisper (tiny.en, int8)",
            "factory": lambda: FasterWhisperEngine(model_name="tiny.en", compute_type="int8"),
        },
        {
            "name": "faster-whisper (tiny.en, float32)",
            "factory": lambda: FasterWhisperEngine(model_name="tiny.en", compute_type="float32"),
        },
        {
            "name": "faster-whisper (base.en, int8)",
            "factory": lambda: FasterWhisperEngine(model_name="base.en", compute_type="int8"),
        },
        {
            "name": "faster-whisper (small.en, int8)",
            "factory": lambda: FasterWhisperEngine(model_name="small.en", compute_type="int8"),
        },
        {
            "name": "openai-whisper (tiny.en, float32)",
            "factory": lambda: OpenAIWhisperEngine(model_name="tiny.en"),
        },
        {
            "name": "openai-whisper (base.en, float32)",
            "factory": lambda: OpenAIWhisperEngine(model_name="base.en"),
        },
    ]

    all_results = []

    for config in configurations:
        cfg_name = config["name"]
        print(f"\n---> Evaluating: {cfg_name}")
        engine = config["factory"]()

        # 1. Measure model load time
        t0 = time.perf_counter()
        engine.load()
        load_time_ms = (time.perf_counter() - t0) * 1000
        print(f"     Model load time: {load_time_ms:.1f} ms")

        # 2. Evaluate each sample
        sample_metrics = []
        total_substitutions = 0
        total_deletions = 0
        total_insertions = 0
        total_ref_words = 0

        total_char_sub = 0
        total_char_del = 0
        total_char_ins = 0
        total_ref_chars = 0

        total_inference_time = 0.0
        peak_ram_mb = 0.0

        for sample in samples:
            sample_id = sample["sample_id"]
            wav_path = sample["normalized_wav_path"]
            reference = sample["reference_transcript"]
            audio_dur = sample["duration_seconds"]

            res = engine.transcribe(wav_path)
            total_inference_time += res.inference_duration_seconds
            if res.peak_memory_mb and res.peak_memory_mb > peak_ram_mb:
                peak_ram_mb = res.peak_memory_mb

            # Word Error Rate
            wer, s, d, i, n_w = calculate_wer(reference, res.text, normalize=True)
            total_substitutions += s
            total_deletions += d
            total_insertions += i
            total_ref_words += n_w

            # Character Error Rate
            cer, cs, cd, ci, n_c = calculate_cer(reference, res.text, normalize=True)
            total_char_sub += cs
            total_char_del += cd
            total_char_ins += ci
            total_ref_chars += n_c

            sample_metrics.append({
                "sample_id": sample_id,
                "duration": audio_dur,
                "inference_time": res.inference_duration_seconds,
                "rtf": res.real_time_factor,
                "wer": round(wer, 4),
                "cer": round(cer, 4),
                "hypothesis": res.text,
                "reference": reference,
            })
            print(f"     [{sample_id}] {audio_dur:.1f}s -> {res.inference_duration_seconds:.2f}s (RTF: {res.real_time_factor:.3f}) | WER: {wer:.1%}")
            print(f"       Hyp: {res.text}")
            print(f"       Ref: {reference}")

        # Overall aggregate metrics
        overall_wer = (total_substitutions + total_deletions + total_insertions) / float(total_ref_words) if total_ref_words > 0 else 0.0
        overall_cer = (total_char_sub + total_char_del + total_char_ins) / float(total_ref_chars) if total_ref_chars > 0 else 0.0
        overall_rtf = total_inference_time / total_audio_duration if total_audio_duration > 0 else 0.0

        print(f"     => OVERALL: WER={overall_wer:.2%} (S={total_substitutions}, D={total_deletions}, I={total_insertions}, N={total_ref_words}) | CER={overall_cer:.2%} | RTF={overall_rtf:.4f} | Peak RAM={peak_ram_mb:.1f}MB")

        all_results.append({
            "configuration": cfg_name,
            "load_time_ms": round(load_time_ms, 2),
            "total_inference_time_s": round(total_inference_time, 3),
            "overall_rtf": round(overall_rtf, 4),
            "overall_wer": round(overall_wer, 4),
            "overall_cer": round(overall_cer, 4),
            "substitutions": total_substitutions,
            "deletions": total_deletions,
            "insertions": total_insertions,
            "ref_words": total_ref_words,
            "peak_ram_mb": round(peak_ram_mb, 2),
            "samples": sample_metrics,
        })

        # Unload model
        engine.unload()

    # Save benchmark report json
    report_path = Path("data/benchmark/asr_benchmark_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved benchmark results to {report_path}")

    # Print summary markdown table
    print("\n" + "=" * 80)
    print("ASR BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Configuration':<35} | {'WER':<8} | {'CER':<8} | {'RTF':<8} | {'Inference':<10} | {'Peak RAM':<10} | {'Load Time':<10}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(f"{r['configuration']:<35} | {r['overall_wer']:<7.2%} | {r['overall_cer']:<7.2%} | {r['overall_rtf']:<7.4f} | {r['total_inference_time_s']:<8.2f}s | {r['peak_ram_mb']:<8.1f}MB | {r['load_time_ms']:<8.1f}ms")


if __name__ == "__main__":
    run_benchmark()
