"""Unit tests for the curator KG summary API router.

Tests endpoint behaviour through the ASGI transport with bypass_auth
enabled (middleware injects a curator user). Verifies F001 envelope
structure, status codes, and pagination.

Role enforcement is tested at the dependency level in test_auth_dependencies.
The router tests verify the happy path with bypass_auth (curator by default).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient

from tabletop_oracle.main import app
from tabletop_oracle.services.kg.summary import (
    DocumentContributionData,
    DocumentSummaryData,
    KGSummaryData,
    KGSummaryService,
    SemanticTypeStat,
)

# ---------------------------------------------------------------------------
# Shared constants and helpers
# ---------------------------------------------------------------------------

GAME_ID = UUID("550e8400-e29b-41d4-a716-446655440000")
DOC_ID = UUID("660e8400-e29b-41d4-a716-446655440001")
NOW = datetime(2026, 3, 14, 12, 30, 0, tzinfo=UTC)


async def _override_db() -> Any:
    """Dependency override yielding a mock session."""
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


def _make_summary() -> KGSummaryData:
    """Build a sample KGSummaryData for testing."""
    return KGSummaryData(
        game_id=GAME_ID,
        concept_count=50,
        association_count=100,
        document_count=2,
        semantic_types=[
            SemanticTypeStat(semantic_type="game_mechanic", count=30),
            SemanticTypeStat(semantic_type="rule", count=20),
        ],
        documents=[
            DocumentSummaryData(
                document_id=DOC_ID,
                document_name="Test Rules",
                document_type="core_rules",
                concepts_contributed=35,
                associations_contributed=60,
            ),
        ],
        last_updated_at=NOW,
    )


def _make_contribution() -> DocumentContributionData:
    """Build a sample DocumentContributionData for testing."""
    return DocumentContributionData(
        document_id=DOC_ID,
        document_name="Test Rules",
        document_type="core_rules",
        expansion_id=None,
        concepts_contributed=35,
        associations_contributed=60,
        conflicts_detected=2,
        last_processed_at=NOW,
    )


# ---------------------------------------------------------------------------
# Tests: GET /knowledge-graph/summary
# ---------------------------------------------------------------------------


class TestGetKGSummary:
    """Tests for GET /games/{game_id}/knowledge-graph/summary."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> Any:
        """Override dependencies and enable bypass_auth for all tests."""
        from tabletop_oracle.api.deps import get_db

        app.dependency_overrides[get_db] = _override_db
        yield
        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_returns_200_with_summary(self) -> None:
        """Summary endpoint returns 200 with correct envelope structure."""
        summary = _make_summary()

        with (
            patch("tabletop_oracle.auth.middleware.settings") as mock_settings,
            patch.object(KGSummaryService, "get_summary", new_callable=AsyncMock) as mock_get,
        ):
            mock_settings.bypass_auth = True
            mock_get.return_value = summary

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    f"/api/v1/games/{GAME_ID}/knowledge-graph/summary",
                    headers={"X-Request-ID": "req-test-1"},
                )

        assert response.status_code == 200
        body = response.json()
        assert "data" in body
        assert "meta" in body
        assert body["meta"]["request_id"] == "req-test-1"

    @pytest.mark.asyncio
    async def test_summary_data_structure(self) -> None:
        """Summary response data matches the design contract."""
        summary = _make_summary()

        with (
            patch("tabletop_oracle.auth.middleware.settings") as mock_settings,
            patch.object(KGSummaryService, "get_summary", new_callable=AsyncMock) as mock_get,
        ):
            mock_settings.bypass_auth = True
            mock_get.return_value = summary

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    f"/api/v1/games/{GAME_ID}/knowledge-graph/summary",
                )

        data = response.json()["data"]
        assert data["game_id"] == str(GAME_ID)
        assert data["concept_count"] == 50
        assert data["association_count"] == 100
        assert data["document_count"] == 2
        assert len(data["semantic_types"]) == 2
        assert data["semantic_types"][0]["type"] == "game_mechanic"
        assert data["semantic_types"][0]["count"] == 30
        assert len(data["documents"]) == 1
        assert data["documents"][0]["document_id"] == str(DOC_ID)
        assert data["documents"][0]["concepts_contributed"] == 35
        assert data["last_updated_at"] is not None


# ---------------------------------------------------------------------------
# Tests: GET /knowledge-graph/documents
# ---------------------------------------------------------------------------


class TestGetKGDocuments:
    """Tests for GET /games/{game_id}/knowledge-graph/documents."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> Any:
        """Override dependencies and enable bypass_auth for all tests."""
        from tabletop_oracle.api.deps import get_db

        app.dependency_overrides[get_db] = _override_db
        yield
        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_returns_200_with_pagination(self) -> None:
        """Documents endpoint returns 200 with paginated envelope."""
        contribution = _make_contribution()

        with (
            patch("tabletop_oracle.auth.middleware.settings") as mock_settings,
            patch.object(
                KGSummaryService,
                "get_document_contributions",
                new_callable=AsyncMock,
            ) as mock_get,
        ):
            mock_settings.bypass_auth = True
            mock_get.return_value = ([contribution], 1)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    f"/api/v1/games/{GAME_ID}/knowledge-graph/documents",
                    headers={"X-Request-ID": "req-test-2"},
                )

        assert response.status_code == 200
        body = response.json()
        assert "data" in body
        assert "meta" in body
        assert body["meta"]["request_id"] == "req-test-2"
        assert "pagination" in body["meta"]
        assert body["meta"]["pagination"]["total_count"] == 1

    @pytest.mark.asyncio
    async def test_document_contribution_structure(self) -> None:
        """Document contribution response matches the design contract."""
        contribution = _make_contribution()

        with (
            patch("tabletop_oracle.auth.middleware.settings") as mock_settings,
            patch.object(
                KGSummaryService,
                "get_document_contributions",
                new_callable=AsyncMock,
            ) as mock_get,
        ):
            mock_settings.bypass_auth = True
            mock_get.return_value = ([contribution], 1)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    f"/api/v1/games/{GAME_ID}/knowledge-graph/documents",
                )

        data = response.json()["data"]
        assert len(data) == 1
        doc = data[0]
        assert doc["document_id"] == str(DOC_ID)
        assert doc["name"] == "Test Rules"
        assert doc["type"] == "core_rules"
        assert doc["expansion_id"] is None
        assert doc["concepts_contributed"] == 35
        assert doc["associations_contributed"] == 60
        assert doc["conflicts_detected"] == 2

    @pytest.mark.asyncio
    async def test_pagination_parameters_forwarded(self) -> None:
        """Pagination query params are forwarded to the service."""
        with (
            patch("tabletop_oracle.auth.middleware.settings") as mock_settings,
            patch.object(
                KGSummaryService,
                "get_document_contributions",
                new_callable=AsyncMock,
            ) as mock_get,
        ):
            mock_settings.bypass_auth = True
            mock_get.return_value = ([], 0)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                await client.get(
                    f"/api/v1/games/{GAME_ID}/knowledge-graph/documents",
                    params={"page": 2, "page_size": 10},
                )

            mock_get.assert_awaited_once()
            call_kwargs = mock_get.call_args
            assert call_kwargs.kwargs["limit"] == 10
            assert call_kwargs.kwargs["offset"] == 10  # (page 2 - 1) * 10


# ---------------------------------------------------------------------------
# Tests: POST /knowledge-graph/reprocess
# ---------------------------------------------------------------------------


class TestReprocessKG:
    """Tests for POST /games/{game_id}/knowledge-graph/reprocess."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> Any:
        """Override dependencies and enable bypass_auth for all tests."""
        from tabletop_oracle.api.deps import get_db

        app.dependency_overrides[get_db] = _override_db
        yield
        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_returns_202_accepted(self) -> None:
        """Reprocess endpoint returns 202 with correct structure."""
        with (
            patch("tabletop_oracle.auth.middleware.settings") as mock_settings,
            patch.object(
                KGSummaryService,
                "count_documents_for_reprocess",
                new_callable=AsyncMock,
            ) as mock_count,
        ):
            mock_settings.bypass_auth = True
            mock_count.return_value = 5

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    f"/api/v1/games/{GAME_ID}/knowledge-graph/reprocess",
                    headers={"X-Request-ID": "req-test-3"},
                )

        assert response.status_code == 202
        body = response.json()
        assert body["data"]["game_id"] == str(GAME_ID)
        assert body["data"]["documents_queued"] == 5
        assert body["data"]["status"] == "reprocessing"
        assert body["meta"]["request_id"] == "req-test-3"

    @pytest.mark.asyncio
    async def test_returns_404_for_empty_graph(self) -> None:
        """Reprocess endpoint returns 404 when no KG data exists."""
        with (
            patch("tabletop_oracle.auth.middleware.settings") as mock_settings,
            patch.object(
                KGSummaryService,
                "count_documents_for_reprocess",
                new_callable=AsyncMock,
            ) as mock_count,
        ):
            mock_settings.bypass_auth = True
            mock_count.return_value = 0

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    f"/api/v1/games/{GAME_ID}/knowledge-graph/reprocess",
                )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_unauthenticated_returns_401(self) -> None:
        """Reprocess endpoint returns 401 without authentication."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                f"/api/v1/games/{GAME_ID}/knowledge-graph/reprocess",
            )

        assert response.status_code == 401
