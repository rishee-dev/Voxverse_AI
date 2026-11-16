"""Common FastAPI dependency providers.

These callables are injected via ``Depends()`` in route handlers to supply
shared resources such as settings and loggers.

Owner: Shared (all team members)
"""

from __future__ import annotations

import logging
from functools import lru_cache

from voxverse.config.settings import Settings
from voxverse.config.settings import get_settings as _get_settings


def get_settings() -> Settings:
    """FastAPI dependency that provides the application settings singleton.

    Usage::

        @router.get("/example")
        async def example(settings: Settings = Depends(get_settings)):
            ...
    """
    return _get_settings()


def get_logger(name: str = "voxverse.api") -> logging.Logger:
    """FastAPI dependency that provides a named logger.

    By default returns the ``voxverse.api`` logger.  Override *name*
    for subsystem-specific loggers.

    Usage::

        @router.get("/example")
        async def example(logger: logging.Logger = Depends(get_logger)):
            logger.info("handling request")
    """
    return logging.getLogger(name)
