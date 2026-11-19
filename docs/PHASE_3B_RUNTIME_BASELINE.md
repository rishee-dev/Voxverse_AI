# VoxVerse — Phase 3B: Preflight Runtime Baseline & Hardware Telemetry

> **Document ID:** `VOX-REP-006`  
> **Phase:** 3B (IndicF5 Controlled TTS Proof of Concept)  
> **Timestamp:** November 2026  
> **Host Environment:** Windows 11 Home ARM64 (Qualcomm Snapdragon X) running Python 3.14.8 AMD64 emulation  
> **Status:** MEASURED & RECORDED  

---

## 1. Host Hardware & Architecture Telemetry

| Parameter | Measured Value | Validation Status |
|:---|:---|:---|
| **Operating System** | `Windows-11-10.0.26300-SP0` (Windows 11 Home) | ✅ VERIFIED |
| **CPU Architecture** | Qualcomm Snapdragon X, ARMv8 Family 8 Model 1 (8 Cores) | ✅ VERIFIED |
| **Host System Architecture** | `ARM64` | ✅ VERIFIED |
| **Python Process Architecture** | `64bit, WindowsPE` (AMD64 running under Windows on ARM emulation) | ✅ VERIFIED |
| **Python Executable** | `C:\Users\kuchi\OneDrive\Documents\VOXVERSE\.venv\Scripts\python.exe` | ✅ VERIFIED |
| **Python Runtime Version** | `3.14.8 (tags/v3.14.8:8e6e75d, Sep 30 2026, 18:19:33) [MSC v.1944 64 bit (AMD64)]` | ✅ VERIFIED |
| **Virtual Environment** | `.venv` Active (Isolated workspace environment) | ✅ VERIFIED |

---

## 2. Memory & Storage Baseline

Measurements captured via `psutil` and `shutil` immediately prior to model preparation:

```
Total Physical RAM:       15.60 GB (16,747,560,960 bytes)
Currently Used RAM:       11.48 GB (73.6%)
Currently Available RAM:   4.12 GB (4,423,733,248 bytes) (26.4%)
Initial Process RSS:      36.46 MB

Drive C: Total Disk Space: 458.00 GB
Drive C: Free Disk Space:  292.26 GB (63.8% free)
```

---

## 3. Process & Model Residency Audit

- **Active VoxVerse Processes:** None. (Audited via process scan; only the telemetry script was executing).
- **Resident ASR Weights (`faster-whisper`):** Not loaded.
- **Resident Translation Weights (`IndicTrans2`):** Not loaded.
- **GPU Accelerator / CUDA:** None present (CPU-only execution).

---

## 4. Memory Admission Policy Evaluation

Under the VoxVerse Phase 3B Memory Admission Policy:
- **Tier 1 (<4.0 GB Available):** Hard stop. No inference permitted.
- **Tier 2 (4.0 – 5.0 GB Available):** **Current State (4.12 GB).** Permits dependency preparation, isolated environment staging, model asset downloads, and loading tests. Full multi-step inference requires explicit memory safety assessment to prevent pagefile thrashing.
- **Tier 3 ($\ge 5.0$ GB Available):** Permits controlled single-request inference.

**Admitted Scope:** Dependency inspection, model asset verification, and isolated model loading validation. Full synthesis inference will be gated upon confirming that free memory remains above safety margins.
