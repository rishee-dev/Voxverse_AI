"""
VoxVerse GenAI ASR — Automatic Speech Recognition (P2).

Owner: P2

Responsibilities:
- English speech-to-text transcription
- Support for multiple ASR backends (Whisper, Google Speech API, etc.)
- Timestamped word/segment-level alignment
- Confidence scoring and transcription quality estimation
- Handling of various audio formats and sampling rates

The ASR module receives preprocessed audio segments from the audio
module (P1) and produces text transcriptions for the translation
module (P2).
"""

__all__: list[str] = []
