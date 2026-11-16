"""
VoxVerse GenAI TTS — Text-to-Speech with voice cloning (P3).

Owner: P3

Responsibilities:
- Telugu text-to-speech synthesis
- Voice cloning using enrolled speaker profiles
- Prosody and intonation control for natural-sounding output
- Support for multiple TTS backends (Coqui TTS, VITS, etc.)
- Audio post-processing (normalization, format conversion)

The TTS module receives Telugu text from the translation module (P2)
and speaker embeddings from the speaker module (P1) to produce
voice-cloned Telugu speech audio.
"""

__all__: list[str] = []
