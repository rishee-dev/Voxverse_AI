"""
VoxVerse — AI-powered English-to-Telugu voice translation platform.

This is the root package for the VoxVerse application. It provides:
- Audio preprocessing and voice activity detection (P1)
- Speaker voice enrollment and profile management (P1)
- Automatic speech recognition (P2)
- English-to-Telugu text translation (P2)
- Text-to-speech synthesis with voice cloning (P3)
- Speaker similarity evaluation and quality metrics (P3)
- End-to-end pipeline orchestration and job processing (P4)

Architecture:
    voxverse/
    ├── api/          — FastAPI application layer (routes, dependencies)
    ├── genai/        — AI/ML components (audio, speaker, ASR, translation, TTS, pipeline, evaluation)
    ├── config/       — Centralized configuration management
    └── storage/      — Storage abstractions for audio and voice profiles
"""

__all__: list[str] = []

__version__ = "0.1.0"
