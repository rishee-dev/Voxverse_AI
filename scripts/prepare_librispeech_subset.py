"""Download and prepare a reproducible subset of LibriSpeech test-clean for ASR benchmarking.

Dataset: LibriSpeech (test-clean split)
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
Source: https://www.openslr.org/12/
Citation: Panayotov, V., Chen, G., Povey, D., & Khudanpur, S. (ICASSP 2015).
"""

from __future__ import annotations

import json
import sys
import tarfile
import urllib.request
from pathlib import Path

# Ensure voxverse is in path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voxverse.genai.audio.normalizer import AudioNormalizer

OPENSLR_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
TARGET_SPEAKER_DIR = "LibriSpeech/test-clean/6930/75918/"
TARGET_SAMPLES = [
    "6930-75918-0000",
    "6930-75918-0001",
    "6930-75918-0002",
    "6930-75918-0003",
    "6930-75918-0004",
]


def prepare_subset() -> Path:
    out_dir = Path("data/benchmark/librispeech").resolve()
    raw_dir = out_dir / "raw"
    wav_dir = out_dir / "wav_16k"
    raw_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        print(f"Benchmark manifest already exists at {manifest_path}")
        return manifest_path

    print(f"Streaming LibriSpeech test-clean from {OPENSLR_URL}...")
    req = urllib.request.Request(OPENSLR_URL, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=60)

    transcripts: dict[str, str] = {}
    extracted_flac: dict[str, Path] = {}

    target_flac_names = {f"{TARGET_SPEAKER_DIR}{s}.flac": s for s in TARGET_SAMPLES}
    target_trans_name = f"{TARGET_SPEAKER_DIR}6930-75918.trans.txt"

    with tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for member in tar:
            if member.name == target_trans_name:
                print(f"Extracting {member.name}...")
                f = tar.extractfile(member)
                if f is not None:
                    for line in f.read().decode("utf-8").splitlines():
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            transcripts[parts[0]] = parts[1]

            elif member.name in target_flac_names:
                sample_id = target_flac_names[member.name]
                print(f"Extracting {member.name} ({member.size} bytes)...")
                flac_path = raw_dir / f"{sample_id}.flac"
                f = tar.extractfile(member)
                if f is not None:
                    flac_path.write_bytes(f.read())
                    extracted_flac[sample_id] = flac_path

            # If we found all target samples and transcript, we can stop streaming
            if len(extracted_flac) == len(TARGET_SAMPLES) and len(transcripts) > 0:
                print("All target samples and transcripts retrieved.")
                break

    # Normalize audio using P1 AudioNormalizer
    print("Normalizing audio files with P1 AudioNormalizer to 16 kHz mono 16-bit PCM WAV...")
    normalizer = AudioNormalizer()
    manifest_records = []

    for sample_id in TARGET_SAMPLES:
        flac_path = extracted_flac[sample_id]
        wav_path = wav_dir / f"{sample_id}.wav"
        dest_path, meta, quality = normalizer.normalize(flac_path, wav_path)

        raw_ref = transcripts.get(sample_id, "")
        manifest_records.append({
            "sample_id": sample_id,
            "raw_flac_path": str(flac_path),
            "normalized_wav_path": str(dest_path),
            "duration_seconds": meta.duration_seconds,
            "sample_rate": meta.normalized_sample_rate,
            "channels": meta.normalized_channels,
            "reference_transcript": raw_ref,
        })
        print(f"[{sample_id}] {meta.duration_seconds:.2f}s | Ref: {raw_ref[:50]}...")

    manifest_data = {
        "dataset_name": "LibriSpeech test-clean (subset)",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "source": "https://www.openslr.org/12/",
        "citation": "Panayotov, V., Chen, G., Povey, D., & Khudanpur, S. (ICASSP 2015).",
        "speaker_id": "6930",
        "chapter_id": "75918",
        "sample_count": len(manifest_records),
        "total_duration_seconds": sum(r["duration_seconds"] for r in manifest_records),
        "samples": manifest_records,
    }

    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    print(f"Manifest written to {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    prepare_subset()
