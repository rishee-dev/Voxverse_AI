"""
VoxVerse Storage — Storage abstractions for audio and voice data.

Provides a storage abstraction layer supporting:
- Local filesystem storage (development)
- Cloud object storage (production: GCS, S3, etc.)
- Audio file management (upload, retrieval, deletion)
- Voice profile persistence (embeddings, metadata)
- Temporary file handling and cleanup

Storage paths are configurable via the config module and default
to the project-level storage/ directory for local development.
"""

__all__: list[str] = []
