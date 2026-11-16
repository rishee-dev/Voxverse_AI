"""
VoxVerse API — FastAPI application layer.

Provides the HTTP API surface for the VoxVerse platform, including:
- REST endpoints for audio upload, translation jobs, and voice management
- WebSocket support for real-time processing status
- Dependency injection for services and configuration
- Request validation and error handling

Sub-packages:
    routes/        — API route handlers organized by domain
    dependencies/  — FastAPI dependency injection providers
"""

from voxverse.api.app import app, create_app

__all__: list[str] = ["app", "create_app"]
