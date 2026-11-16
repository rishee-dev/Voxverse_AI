"""
VoxVerse GenAI Pipeline — Orchestration and job processing (P4).

Owner: P4

Responsibilities:
- End-to-end translation pipeline orchestration
- Job queue management and async task processing
- Pipeline stage coordination (audio → ASR → translation → TTS)
- Error handling, retries, and graceful degradation
- Progress tracking and status reporting
- Resource management and concurrency control

The pipeline module coordinates all other GenAI modules to provide
a seamless voice translation workflow from English audio input to
Telugu voice-cloned audio output.
"""

__all__: list[str] = []
