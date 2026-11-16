"""Structured logging configuration for VoxVerse.

Provides both human-readable (``text``) and machine-parseable (``json``)
log formats.  A ``request_id`` field is injected via a context-var filter
so that every log record emitted during a request carries the same
correlation identifier.

Owner: Shared (all team members)
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

# ── Context variable for per-request correlation ────────────────────────

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


def set_request_id(request_id: str) -> None:
    """Set the current request ID in the context variable.

    Call this at the start of each request (e.g. in middleware) so that
    all subsequent log records include the correlation ID.
    """
    request_id_ctx.set(request_id)


def get_request_id() -> str:
    """Return the current request ID from the context variable."""
    return request_id_ctx.get()


# ── Custom filter that injects request_id ───────────────────────────────


class RequestIdFilter(logging.Filter):
    """Logging filter that attaches ``request_id`` to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add the contextvar ``request_id`` to the log record."""
        record.request_id = request_id_ctx.get()  # type: ignore[attr-defined]
        return True


# ── JSON formatter ──────────────────────────────────────────────────────


class JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line.

    Fields: ``timestamp``, ``level``, ``logger``, ``message``,
    ``request_id``, and any extra ``exc_info``.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format *record* as a JSON string."""
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[1] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)


# ── Text formatter ──────────────────────────────────────────────────────

_TEXT_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "req=%(request_id)s | %(message)s"
)
_TEXT_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


# ── Public setup function ──────────────────────────────────────────────


def setup_logging(
    *,
    level: str = "INFO",
    log_format: str = "text",
    logger_name: str = "voxverse",
) -> logging.Logger:
    """Configure and return the application root logger.

    Parameters
    ----------
    level:
        Minimum severity (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_format:
        ``"text"`` for human-readable output, ``"json"`` for structured
        JSON lines (recommended in production / container environments).
    logger_name:
        Name of the logger to configure. Defaults to ``"voxverse"``.

    Returns
    -------
    logging.Logger
        The configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Prevent duplicate handlers when called more than once (e.g. tests).
    if logger.handlers:
        logger.handlers.clear()

    handler = logging.StreamHandler(stream=sys.stdout)

    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(fmt=_TEXT_FORMAT, datefmt=_TEXT_DATE_FORMAT)
        )

    handler.addFilter(RequestIdFilter())
    logger.addHandler(handler)

    # Don't propagate to the root logger to avoid duplicate output.
    logger.propagate = False

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the ``voxverse`` namespace.

    Parameters
    ----------
    name:
        Optional child name. ``get_logger("api")`` returns the logger
        ``voxverse.api``.  If *None*, returns the ``voxverse`` logger.
    """
    base = "voxverse"
    if name:
        return logging.getLogger(f"{base}.{name}")
    return logging.getLogger(base)
