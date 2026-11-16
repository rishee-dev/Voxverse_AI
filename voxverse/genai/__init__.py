"""
VoxVerse GenAI — AI/ML components for voice translation.

This package contains all AI and machine learning modules powering the
English-to-Telugu voice translation pipeline:

Sub-packages:
    audio/       — (P1) Audio preprocessing, VAD, quality validation
    speaker/     — (P1) Voice enrollment, profile storage, speaker embeddings
    asr/         — (P2) Automatic speech recognition (English)
    translation/ — (P2) English-to-Telugu text translation
    tts/         — (P3) Text-to-speech synthesis with voice cloning
    evaluation/  — (P3) Speaker similarity scoring and quality metrics
    pipeline/    — (P4) End-to-end orchestration and job processing
"""

__all__: list[str] = []
