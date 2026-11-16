"""
VoxVerse Integration Tests — Tests requiring external services or models.

Integration tests may depend on:
- FFmpeg being installed and available on PATH
- ML model weights being downloaded
- External API keys being configured
- The full FastAPI application stack

Mark slow or resource-intensive tests with ``@pytest.mark.slow`` so
they can be excluded from quick CI runs.
"""

__all__: list[str] = []
