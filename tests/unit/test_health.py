"""Unit tests for the GET /health endpoint.

Covers:
- HTTP 200 OK status code
- Presence of required fields (status, version, timestamp)
- "healthy" status value
- ISO-8601 formatted UTC timestamp validation
- Version alignment with configuration
- Correlation and timing response headers
- Method validation (POST disallowed)

Owner: P4 (Pipeline Engineer) / Shared
"""

from __future__ import annotations

from datetime import datetime

import pytest
from httpx import ASGITransport, AsyncClient

from voxverse.api.app import app
from voxverse.config.settings import get_settings


@pytest.mark.asyncio
async def test_health_returns_status_200(async_client: AsyncClient) -> None:
    """Test that GET /health returns HTTP 200 OK.

    Arrange: Async HTTP client connected to the application.
    Act: Send GET request to /health.
    Assert: HTTP status code is 200.
    """
    # Arrange
    endpoint = "/health"

    # Act
    response = await async_client.get(endpoint)

    # Assert
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_response_has_required_fields(async_client: AsyncClient) -> None:
    """Test that GET /health response contains all required schema fields.

    Arrange: Async HTTP client connected to the application.
    Act: Send GET request to /health.
    Assert: JSON body contains 'status', 'version', and 'timestamp'.
    """
    # Arrange
    required_fields = {"status", "version", "timestamp"}

    # Act
    response = await async_client.get("/health")
    payload = response.json()

    # Assert
    assert required_fields.issubset(payload.keys())


@pytest.mark.asyncio
async def test_health_status_is_healthy(async_client: AsyncClient) -> None:
    """Test that GET /health reports status as 'healthy'.

    Arrange: Async HTTP client connected to the application.
    Act: Send GET request to /health.
    Assert: 'status' field equals literal string 'healthy'.
    """
    # Arrange
    expected_status = "healthy"

    # Act
    response = await async_client.get("/health")
    data = response.json()

    # Assert
    assert data["status"] == expected_status


@pytest.mark.asyncio
async def test_health_timestamp_is_valid_iso_format(async_client: AsyncClient) -> None:
    """Test that the timestamp returned by GET /health is valid ISO-8601 format.

    Arrange: Async HTTP client connected to the application.
    Act: Send GET request to /health and extract the timestamp string.
    Assert: String can be parsed into a valid datetime object with timezone info.
    """
    # Arrange & Act
    response = await async_client.get("/health")
    timestamp_str = response.json()["timestamp"]

    # Assert
    # Handles both 'Z' and offset-based '+00:00' UTC representations
    normalized_str = timestamp_str.replace("Z", "+00:00")
    parsed_dt = datetime.fromisoformat(normalized_str)
    assert parsed_dt is not None
    assert parsed_dt.tzinfo is not None


@pytest.mark.asyncio
async def test_health_version_matches_application_version(async_client: AsyncClient) -> None:
    """Test that GET /health reports the version matching app settings.

    Arrange: Retrieve expected version from current application settings.
    Act: Send GET request to /health.
    Assert: 'version' in response matches the settings version string.
    """
    # Arrange
    settings = get_settings()
    expected_version = settings.app_version

    # Act
    response = await async_client.get("/health")
    data = response.json()

    # Assert
    assert data["version"] == expected_version


@pytest.mark.asyncio
async def test_health_response_includes_correlation_headers(async_client: AsyncClient) -> None:
    """Test that GET /health includes X-Request-ID and X-Process-Time-Ms headers.

    Arrange: Async HTTP client connected to the application.
    Act: Send GET request to /health.
    Assert: 'x-request-id' and 'x-process-time-ms' response headers are present.
    """
    # Arrange & Act
    response = await async_client.get("/health")

    # Assert
    assert "x-request-id" in response.headers
    assert "x-process-time-ms" in response.headers
    assert len(response.headers["x-request-id"]) > 0


@pytest.mark.asyncio
async def test_health_method_not_allowed_for_post(async_client: AsyncClient) -> None:
    """Test that POST /health is rejected with HTTP 405 Method Not Allowed.

    Arrange: Async HTTP client connected to the application.
    Act: Send POST request to /health.
    Assert: HTTP status code is 405.
    """
    # Arrange & Act
    response = await async_client.post("/health", json={})

    # Assert
    assert response.status_code == 405


@pytest.mark.asyncio
async def test_health_with_direct_transport_client() -> None:
    """Test GET /health using an explicit AsyncClient with ASGITransport.

    Verifies the standalone client construction pattern required by the specification.

    Arrange: Explicitly build AsyncClient using ASGITransport and base_url='http://test'.
    Act: Send GET request to /health.
    Assert: HTTP status code is 200 and status is 'healthy'.
    """
    # Arrange
    transport = ASGITransport(app=app)

    # Act
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    # Assert
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
