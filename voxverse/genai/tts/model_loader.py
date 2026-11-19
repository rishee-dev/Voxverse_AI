"""Isolated model loader with memory admission gates and sequential lifecycle safety.

Owner: P3 — Voice Generation Engineer
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any
import psutil

from voxverse.genai.tts.config import TTSConfig
from voxverse.genai.tts.exceptions import (
    GatedRepositoryError,
    MemoryAdmissionError,
    MissingCheckpointError,
    MissingVocoderError,
    ModelLoadError,
)

logger = logging.getLogger(__name__)


class TTSModelLoader:
    """Safe model loader that enforces preflight memory checks and sequential residency."""

    def __init__(self, config: TTSConfig | None = None) -> None:
        self.config = config or TTSConfig()

    def check_memory_admission(self, strict: bool = True) -> float:
        """Verify that the host has sufficient available physical RAM before loading models.

        Parameters
        ----------
        strict:
            If True, enforces the hard-stop minimum threshold (`memory_min_available_gb`).

        Returns
        -------
        float
            Current available physical RAM in GB.

        Raises
        ------
        MemoryAdmissionError
            If available physical RAM is below the admission threshold.
        """
        status = self.config.check_memory_admission()
        if strict and not status.admitted:
            logger.error("Memory admission check failed: %s", status.message)
            raise MemoryAdmissionError(
                available_gb=status.available_ram_gb,
                required_gb=self.config.memory_min_available_gb,
            )
        return status.available_ram_gb

    def cleanup_memory(self) -> float:
        """Run thorough garbage collection and return the updated resident process memory in MB."""
        gc.collect()
        proc = psutil.Process()
        rss_mb = proc.memory_info().rss / (1024**2)
        logger.debug("Memory cleaned up via gc.collect(); process RSS is now %.2f MB", rss_mb)
        return rss_mb

    def resolve_checkpoint_path(self, model_id: str | None = None) -> Path:
        """Verify the presence of the TTS model checkpoint in the local cache or repository.

        Parameters
        ----------
        model_id:
            Hugging Face model identifier or local path.

        Returns
        -------
        Path
            Path to verified model checkpoint weights.

        Raises
        ------
        GatedRepositoryError
            If accessing the upstream Hugging Face repo fails due to gated authentication requirements.
        MissingCheckpointError
            If checkpoint file cannot be found locally.
        """
        target_model = model_id or self.config.model_id
        cache_dir = Path(self.config.model_cache_dir)
        local_safetensors = cache_dir / "model.safetensors"

        if local_safetensors.exists() and local_safetensors.is_file():
            return local_safetensors

        # Attempt to inspect Hugging Face hub without unprompted mass download
        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import GatedRepoError
        except ImportError as exc:
            raise ModelLoadError(f"huggingface_hub is required to resolve checkpoint: {exc}") from exc

        try:
            # Check for config.json as indicator of repository accessibility
            resolved = hf_hub_download(
                repo_id=target_model,
                filename="model.safetensors",
                local_files_only=True,
            )
            return Path(resolved)
        except GatedRepoError as exc:
            logger.error("Repository '%s' is gated and requires authentication: %s", target_model, exc)
            raise GatedRepositoryError(repo_id=target_model) from exc
        except Exception:
            # If not cached locally, check if online download is permitted or blocked by gating
            try:
                # Test access with a tiny metadata file first
                hf_hub_download(repo_id=target_model, filename="config.json")
            except GatedRepoError as exc:
                raise GatedRepositoryError(repo_id=target_model) from exc
            except Exception:
                pass

            raise MissingCheckpointError(
                checkpoint_path=str(local_safetensors),
                model_id=target_model,
            )

    def resolve_vocoder_path(self, vocoder_id: str | None = None) -> Path:
        """Verify the presence of neural vocoder checkpoint.

        Parameters
        ----------
        vocoder_id:
            Vocoder repository ID or local path.

        Returns
        -------
        Path
            Path to verified vocoder config or weights.

        Raises
        ------
        MissingVocoderError
            If vocoder assets cannot be resolved.
        """
        target_vocoder = vocoder_id or self.config.vocoder_id
        try:
            from huggingface_hub import hf_hub_download
            resolved = hf_hub_download(repo_id=target_vocoder, filename="config.yaml")
            return Path(resolved)
        except Exception as exc:
            raise MissingVocoderError(vocoder_id=target_vocoder) from exc
