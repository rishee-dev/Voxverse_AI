"""Configuration settings, thresholds, and memory policies for TTS and voice generation.

Owner: P3 — Voice Generation Engineer
"""

from __future__ import annotations

import psutil
from pydantic import BaseModel, Field


class MemoryAdmissionStatus(BaseModel):
    """Result of an evaluation against the host memory admission policy."""

    available_ram_gb: float = Field(..., description="Available physical memory in GB.")
    total_ram_gb: float = Field(..., description="Total installed physical memory in GB.")
    used_ram_percent: float = Field(..., description="Percentage of physical memory used.")
    admitted: bool = Field(..., description="Whether inference is permitted under safety policy.")
    tier: str = Field(..., description="Safety tier: 'TIER_1_BLOCKED', 'TIER_2_RESTRICTED', or 'TIER_3_ADMITTED'.")
    message: str = Field(..., description="Human-readable decision explanation.")


class TTSConfig(BaseModel):
    """Runtime configuration for the Text-to-Speech synthesis subsystem."""

    model_id: str = Field(
        default="ai4bharat/IndicF5",
        description="Hugging Face repository ID for the primary TTS model.",
    )
    vocoder_id: str = Field(
        default="charactr/vocos-mel-24khz",
        description="Hugging Face repository ID for the neural vocoder.",
    )
    device: str = Field(
        default="cpu",
        description="Inference device ('cpu'). GPU/CUDA is prohibited on this deployment.",
    )
    nfe_steps: int = Field(
        default=16,
        ge=8,
        le=64,
        description="Number of flow-matching function evaluation steps (ODE solver steps).",
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speaking rate multiplier.",
    )
    output_sample_rate: int = Field(
        default=24000,
        description="Output audio sampling frequency in Hz.",
    )
    max_text_characters: int = Field(
        default=500,
        ge=1,
        le=2000,
        description="Maximum length of target Telugu text per synthesis request.",
    )
    min_ref_audio_duration_seconds: float = Field(
        default=2.0,
        ge=0.5,
        description="Minimum reference audio duration required for voice conditioning.",
    )
    max_ref_audio_duration_seconds: float = Field(
        default=15.0,
        le=30.0,
        description="Maximum permitted reference audio duration to prevent memory spikes.",
    )
    inference_timeout_seconds: float = Field(
        default=60.0,
        ge=5.0,
        le=300.0,
        description="Maximum execution wall-clock time allowed before aborting synthesis.",
    )
    memory_min_available_gb: float = Field(
        default=4.0,
        description="Minimum physical RAM (GB) required to attempt inference. Hard stop if below.",
    )
    memory_safe_available_gb: float = Field(
        default=5.0,
        description="Target physical RAM (GB) required for unrestricted inference attempts.",
    )
    model_cache_dir: str = Field(
        default="models/tts",
        description="Filesystem directory for caching downloaded model checkpoints.",
    )

    def check_memory_admission(self) -> MemoryAdmissionStatus:
        """Evaluate the live host memory against the defined admission policy."""
        vm = psutil.virtual_memory()
        available_gb = vm.available / (1024**3)
        total_gb = vm.total / (1024**3)
        used_percent = vm.percent

        if available_gb < self.memory_min_available_gb:
            return MemoryAdmissionStatus(
                available_ram_gb=round(available_gb, 2),
                total_ram_gb=round(total_gb, 2),
                used_ram_percent=round(used_percent, 1),
                admitted=False,
                tier="TIER_1_BLOCKED",
                message=(
                    f"Blocked: available physical RAM is {available_gb:.2f} GB, which is below "
                    f"the minimum threshold of {self.memory_min_available_gb:.2f} GB. "
                    "Inference stopped to protect system stability."
                ),
            )
        elif available_gb < self.memory_safe_available_gb:
            return MemoryAdmissionStatus(
                available_ram_gb=round(available_gb, 2),
                total_ram_gb=round(total_gb, 2),
                used_ram_percent=round(used_percent, 1),
                admitted=False,
                tier="TIER_2_RESTRICTED",
                message=(
                    f"Restricted: available physical RAM is {available_gb:.2f} GB (between "
                    f"{self.memory_min_available_gb:.2f} GB and {self.memory_safe_available_gb:.2f} GB). "
                    "Preparation and loading tests allowed, but full synthesis requires safe memory margin."
                ),
            )
        else:
            return MemoryAdmissionStatus(
                available_ram_gb=round(available_gb, 2),
                total_ram_gb=round(total_gb, 2),
                used_ram_percent=round(used_percent, 1),
                admitted=True,
                tier="TIER_3_ADMITTED",
                message=(
                    f"Admitted: available physical RAM is {available_gb:.2f} GB, satisfying the "
                    f"safe threshold of {self.memory_safe_available_gb:.2f} GB."
                ),
            )
