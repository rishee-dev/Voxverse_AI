"""VoxVerse API Routes — HTTP endpoint handlers.

Contains FastAPI router modules organized by domain:
- Health check and system status endpoints (health, info)
- Voice profile enrollment, retrieval, and deletion lifecycle (voices)
"""

from voxverse.api.routes import health, info, voices

__all__ = ["health", "info", "voices"]
