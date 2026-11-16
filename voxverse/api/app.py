"""FastAPI application factory for VoxVerse.

Call :func:`create_app` to build a fully-configured ``FastAPI`` instance
with middleware, exception handlers, and all route routers registered.

Owner: P4 — API Integration & System Architecture
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from voxverse import __version__
from voxverse.api.routes import health, info
from voxverse.api.schemas import ErrorResponse
from voxverse.config.logging import get_logger, set_request_id, setup_logging
from voxverse.config.settings import get_settings

logger = get_logger("api")


# ── Exception handlers ──────────────────────────────────────────────────


async def _validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Return a structured 422 response for validation failures."""
    from voxverse.api.schemas import ErrorDetail
    from voxverse.config.logging import get_request_id

    details = [
        ErrorDetail(
            field=" → ".join(str(loc) for loc in err.get("loc", [])),
            message=err.get("msg", "Validation error"),
        )
        for err in exc.errors()
    ]
    body = ErrorResponse(
        error="validation_error",
        message="Request validation failed.",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        details=details,
        request_id=get_request_id(),
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=body.model_dump(mode="json"),
    )


async def _generic_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Catch-all handler that prevents stack traces from leaking."""
    from voxverse.config.logging import get_request_id

    logger.exception("Unhandled exception during request processing")
    body = ErrorResponse(
        error="internal_server_error",
        message="An unexpected error occurred. Please try again later.",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        request_id=get_request_id(),
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=body.model_dump(mode="json"),
    )


async def _http_404_handler(
    request: Request,
    exc: Any,
) -> JSONResponse:
    """Return a structured 404 response for unknown routes."""
    from voxverse.config.logging import get_request_id

    body = ErrorResponse(
        error="not_found",
        message=f"The requested path '{request.url.path}' was not found.",
        status_code=status.HTTP_404_NOT_FOUND,
        request_id=get_request_id(),
    )
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=body.model_dump(mode="json"),
    )


# ── Application factory ────────────────────────────────────────────────


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application.

    This factory:

    1. Reads settings from environment / ``.env``.
    2. Initialises structured logging.
    3. Creates the ``FastAPI`` instance with OpenAPI metadata.
    4. Registers CORS and request-timing middleware.
    5. Registers exception handlers.
    6. Includes all route routers.
    7. Adds startup / shutdown lifecycle hooks.
    """
    settings = get_settings()

    # ── Logging ──────────────────────────────────────────────────────
    setup_logging(
        level=settings.log_level.value,
        log_format=settings.log_format,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Manage application startup and shutdown lifecycle."""
        logger.info(
            "VoxVerse %s starting (env=%s, debug=%s)",
            settings.app_version,
            settings.environment.value,
            settings.debug,
        )
        settings.ensure_directories()
        logger.info("Working directories verified/created.")
        yield
        logger.info("VoxVerse shutting down.")

    # ── App instance ─────────────────────────────────────────────────
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "AI-powered English-to-Telugu voice translation platform. "
            "Upload English audio and receive natural Telugu speech output "
            "with preserved speaker characteristics."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS middleware ──────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )

    # ── Request timing / correlation middleware ──────────────────────
    @app.middleware("http")
    async def request_context_middleware(
        request: Request,
        call_next: Any,
    ) -> Any:
        """Assign a unique request ID and measure response time."""
        request_id = request.headers.get(
            "X-Request-ID",
            uuid.uuid4().hex[:16],
        )
        set_request_id(request_id)

        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"

        logger.info(
            "%s %s → %s (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    # ── Exception handlers ───────────────────────────────────────────
    app.add_exception_handler(
        RequestValidationError,
        _validation_error_handler,  # type: ignore[arg-type]
    )
    app.add_exception_handler(
        status.HTTP_404_NOT_FOUND,
        _http_404_handler,  # type: ignore[arg-type]
    )
    app.add_exception_handler(
        Exception,
        _generic_exception_handler,  # type: ignore[arg-type]
    )

    # ── Routers ──────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(info.router)

    return app


# Default application instance for ASGI servers (e.g. Uvicorn) and testing
app = create_app()

