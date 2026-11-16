"""
VoxVerse GenAI Speaker — Voice enrollment and profile management (P1).

Owner: P1

Responsibilities:
- Speaker voice enrollment from validated reference audio
- Explicit consent verification and audit metadata recording
- Speaker representation extraction for voice conditioning
- Secure voice profile storage and retrieval
- Voice profile deletion lifecycle support
- Embedding similarity computation for speaker verification

Voice profiles are stored alongside consent records and technical metadata
(sample rate, duration, enrollment timestamp, audio quality assessment).
Never commit or expose unauthorized speaker representations.
"""

__all__: list[str] = []
