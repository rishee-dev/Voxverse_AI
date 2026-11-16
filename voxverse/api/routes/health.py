"""Health-check endpoint.

Provides a lightweight probe for load balancers, container orchestrators,
and uptime monitors.

Owner: Shared (all team members)
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from voxverse.api.dependencies.common import get_settings
from voxverse.api.schemas import HealthResponse
from voxverse.config.settings import Settings

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description="Returns the current health status, version, and server timestamp.",
)
async def health_check(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Return a simple health payload.

    This endpoint is intentionally free of authentication and heavy
    processing so that probes receive a fast response.
    """
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        timestamp=datetime.now(tz=timezone.utc),
    )
