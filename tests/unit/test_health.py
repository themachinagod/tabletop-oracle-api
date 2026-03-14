"""Unit tests for the health check endpoint.

Tests cover:
- Healthy response when database is reachable
- Unhealthy (503) response when database is unreachable
- Response envelope structure (DataEnvelope with meta.request_id)
- HealthCheckResult helper behaviour
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from tabletop_oracle.api.health import HealthCheckResult
from tabletop_oracle.main import app

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


def _mock_session_healthy() -> AsyncMock:
    """Create a mock DB session whose execute succeeds."""
    session = AsyncMock()
    session.execute = AsyncMock(return_value=None)
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


def _mock_session_unhealthy() -> AsyncMock:
    """Create a mock DB session whose execute raises."""
    session = AsyncMock()
    session.execute = AsyncMock(side_effect=ConnectionError("connection refused"))
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


async def _override_db_healthy() -> AsyncGenerator[AsyncMock, None]:
    """Dependency override yielding a healthy mock session."""
    session = _mock_session_healthy()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise


async def _override_db_unhealthy() -> AsyncGenerator[AsyncMock, None]:
    """Dependency override yielding an unhealthy mock session."""
    session = _mock_session_unhealthy()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise


class TestHealthEndpointHealthy:
    """Tests for the healthy database scenario."""

    @pytest.fixture(autouse=True)
    def _setup_healthy_db(self) -> Any:
        """Override get_db to return a healthy mock session."""
        from tabletop_oracle.api.deps import get_db

        app.dependency_overrides[get_db] = _override_db_healthy
        yield
        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_health_returns_200(self) -> None:
        """Health endpoint returns 200 when database is reachable."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_returns_healthy_status(self) -> None:
        """Health endpoint data.status is 'healthy' when DB is up."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/health")
        body = response.json()
        assert body["data"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_returns_database_check(self) -> None:
        """Health endpoint includes database check in checks map."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/health")
        body = response.json()
        assert body["data"]["checks"]["database"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_returns_data_envelope(self) -> None:
        """Health response follows DataEnvelope format with meta.request_id."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/api/v1/health",
                headers={"X-Request-ID": "test-req-123"},
            )
        body = response.json()
        assert "data" in body
        assert "meta" in body
        assert body["meta"]["request_id"] == "test-req-123"


class TestHealthEndpointUnhealthy:
    """Tests for the unhealthy database scenario."""

    @pytest.fixture(autouse=True)
    def _setup_unhealthy_db(self) -> Any:
        """Override get_db to return an unhealthy mock session."""
        from tabletop_oracle.api.deps import get_db

        app.dependency_overrides[get_db] = _override_db_unhealthy
        yield
        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_health_returns_503_when_db_down(self) -> None:
        """Health endpoint returns 503 when database is unreachable."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/health")
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_health_returns_unhealthy_status_when_db_down(self) -> None:
        """Health endpoint data.status is 'unhealthy' when DB is down."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/health")
        body = response.json()
        assert body["data"]["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_health_returns_database_unhealthy_when_db_down(self) -> None:
        """Health endpoint checks.database is 'unhealthy' when DB is down."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/health")
        body = response.json()
        assert body["data"]["checks"]["database"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_health_returns_envelope_when_unhealthy(self) -> None:
        """Unhealthy response still follows DataEnvelope format."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/api/v1/health",
                headers={"X-Request-ID": "unhealthy-req-456"},
            )
        body = response.json()
        assert "data" in body
        assert "meta" in body
        assert body["meta"]["request_id"] == "unhealthy-req-456"


class TestHealthCheckResult:
    """Unit tests for the HealthCheckResult helper class."""

    def test_all_healthy(self) -> None:
        """is_healthy returns True when all checks pass."""
        result = HealthCheckResult()
        result.checks["database"] = "healthy"
        assert result.is_healthy is True
        assert result.overall_status == "healthy"

    def test_any_unhealthy(self) -> None:
        """is_healthy returns False when any check fails."""
        result = HealthCheckResult()
        result.checks["database"] = "unhealthy"
        assert result.is_healthy is False
        assert result.overall_status == "unhealthy"

    def test_empty_checks_is_healthy(self) -> None:
        """An empty checks map is considered healthy (vacuously true)."""
        result = HealthCheckResult()
        assert result.is_healthy is True

    def test_to_dict_shape(self) -> None:
        """to_dict returns expected structure."""
        result = HealthCheckResult()
        result.checks["database"] = "healthy"
        d = result.to_dict()
        assert d == {"status": "healthy", "checks": {"database": "healthy"}}

    def test_mixed_checks(self) -> None:
        """Multiple checks with one unhealthy marks overall as unhealthy."""
        result = HealthCheckResult()
        result.checks["database"] = "healthy"
        result.checks["cache"] = "unhealthy"
        assert result.is_healthy is False
        assert result.overall_status == "unhealthy"
