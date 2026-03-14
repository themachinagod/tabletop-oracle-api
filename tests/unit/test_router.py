"""Unit tests for the base API router structure.

Verifies that the api_router correctly includes sub-routers
and that the main app mounts the api_router at /api/v1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from tabletop_oracle.api.router import api_router
from tabletop_oracle.main import app

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


async def _override_db_healthy() -> AsyncGenerator[AsyncMock, None]:
    """Dependency override yielding a healthy mock session."""
    session = AsyncMock()
    session.execute = AsyncMock(return_value=None)
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise


class TestApiRouterStructure:
    """Tests for the api_router configuration."""

    def test_api_router_includes_health(self) -> None:
        """Health router is included in the api_router."""
        paths = [route.path for route in api_router.routes]
        assert "/health" in paths

    def test_health_endpoint_accessible_via_api_v1(self) -> None:
        """Health endpoint is accessible at /api/v1/health through the app."""
        full_paths = []
        for route in app.routes:  # type: ignore[union-attr]
            if hasattr(route, "path"):
                full_paths.append(route.path)
        assert "/api/v1/health" in full_paths


class TestApiRouterIntegration:
    """Integration tests verifying the router works through the app."""

    @pytest.fixture(autouse=True)
    def _setup_db(self) -> Any:
        """Override get_db for router integration tests."""
        from tabletop_oracle.api.deps import get_db

        app.dependency_overrides[get_db] = _override_db_healthy
        yield
        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_health_via_app(self) -> None:
        """Health endpoint responds through the full app stack."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["data"]["status"] == "healthy"
