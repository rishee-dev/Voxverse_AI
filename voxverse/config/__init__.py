"""
VoxVerse Config — Centralized configuration management.

Provides a unified configuration system for the VoxVerse platform:
- Environment-based settings (development, staging, production)
- Pydantic-based configuration validation
- Secrets management (API keys, credentials)
- Feature flags and runtime toggles
- Logging and observability configuration

Configuration is loaded from environment variables and .env files,
with sensible defaults for local development.
"""

__all__: list[str] = []
