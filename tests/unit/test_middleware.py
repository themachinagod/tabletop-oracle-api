"""Unit tests for middleware modules."""

import pytest
from httpx import ASGITransport, AsyncClient

from tabletop_oracle.main import app


@pytest.mark.asyncio
async def test_correlation_id_generated() -> None:
    """Requests without X-Request-ID get one generated."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/health")
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 0


@pytest.mark.asyncio
async def test_correlation_id_forwarded() -> None:
    """Requests with X-Request-ID get the same ID echoed back."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/api/v1/health",
            headers={"X-Request-ID": "test-id-123"},
        )
    assert response.headers["X-Request-ID"] == "test-id-123"
