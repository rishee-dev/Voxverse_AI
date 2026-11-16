"""VoxVerse Test Configuration — Shared pytest fixtures.

Provides reusable fixtures for the entire test suite:
- FastAPI async test client (httpx.AsyncClient with ASGITransport)
- Application instance fixture
- Environment and settings isolation fixtures
- Temporary directory management
- Sample audio configuration defaults

Owner: P4 (Frontend & Integration Engineer) / Shared
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from voxverse.api.app import app
from voxverse.config.settings import Settings, get_settings, reset_settings


@pytest.fixture(autouse=True)
def isolate_settings() -> Generator[None, None, None]:
    """Ensure each test runs with a pristine settings singleton state.

    Resets the cached settings singleton before and after every test
    to prevent cross-test state leakage.
    """
    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def test_app():
    """Return the FastAPI application instance."""
    return app


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Yield an httpx.AsyncClient wired to the FastAPI application.

    Uses ASGITransport to invoke the FastAPI ASGI app directly in-process
    without binding to an actual network port.

    Usage::

        @pytest.mark.asyncio
        async def test_endpoint(async_client: AsyncClient):
            response = await async_client.get("/health")
            assert response.status_code == 200
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test file operations.

    Convenience wrapper around pytest's built-in ``tmp_path`` fixture.

    Returns:
        A Path pointing to a unique temporary directory cleaned up automatically.
    """
    return tmp_path


@pytest.fixture
def sample_audio_config() -> dict[str, int | str | float]:
    """Return a default audio configuration dictionary for tests.

    Provides sensible defaults matching the VoxVerse audio processing pipeline:
        - sample_rate: 16000 (16 kHz, standard for speech models)
        - channels: 1 (mono)
        - format: "wav" (lossless for processing)
        - bit_depth: 16 (16-bit PCM)
        - max_duration_sec: 300 (5-minute cap)
        - min_duration_sec: 0.5 (reject ultra-short clips)
        - target_lufs: -23.0 (EBU R 128 loudness target)
    """
    return {
        "sample_rate": 16000,
        "channels": 1,
        "format": "wav",
        "bit_depth": 16,
        "max_duration_sec": 300,
        "min_duration_sec": 0.5,
        "target_lufs": -23.0,
    }
