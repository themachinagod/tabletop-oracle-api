"""Integration test: health endpoint against a real PostgreSQL database.

Verifies the health endpoint returns the correct envelope format
when connected to an actual database instance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from httpx import ASGITransport, AsyncClient

from tabletop_oracle.api.deps import get_db
from tabletop_oracle.main import app

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.ext.asyncio import AsyncSession


@pytest.mark.integration
class TestHealthIntegration:
    """Integration tests for the health endpoint with a real database."""

    @pytest.mark.asyncio
    async def test_health_returns_healthy_with_real_db(
        self,
        db_session: AsyncSession,
    ) -> None:
        """Health endpoint returns 200 with correct format against real DB.

        The db_session fixture provides a real PostgreSQL connection via
        testcontainers (or CI service).
        """

        async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
            yield db_session

        app.dependency_overrides[get_db] = _override_get_db
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    "/api/v1/health",
                    headers={"X-Request-ID": "integration-test-001"},
                )

            assert response.status_code == 200
            body = response.json()

            # Verify DataEnvelope structure
            assert "data" in body
            assert "meta" in body
            assert body["meta"]["request_id"] == "integration-test-001"

            # Verify health data
            assert body["data"]["status"] == "healthy"
            assert body["data"]["checks"]["database"] == "healthy"
        finally:
            app.dependency_overrides.clear()
