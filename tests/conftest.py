"""Shared test fixtures.

Unit test fixtures (HTTPX async client) are available to all tests.
Integration test fixtures (PostgreSQL via testcontainers) are in
tests/integration/conftest.py.
"""

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient

from tabletop_oracle.main import app


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client for unit/API tests."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
