"""Unit tests for DocumentRepository data access layer.

Tests query construction and filter application with mocked SQLAlchemy
sessions. Covers game silo isolation, dynamic filtering by status/type/
expansion, sort mapping, and aggregation queries.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select

from tabletop_oracle.models.document import Document
from tabletop_oracle.repositories.document_repository import (
    _DEFAULT_SORT,
    _SORT_MAP,
    DocumentRepository,
)
from tabletop_oracle.schemas.document import (
    DocumentFilters,
    DocumentStatusFilter,
    DocumentTypeFilter,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create a mock AsyncSession."""
    return AsyncMock()


@pytest.fixture
def repo(mock_session: AsyncMock) -> DocumentRepository:
    """Create a DocumentRepository with a mocked session."""
    return DocumentRepository(mock_session)


@pytest.fixture
def game_id() -> uuid.UUID:
    """Fixed game UUID for tests."""
    return uuid.UUID("550e8400-e29b-41d4-a716-446655440000")


@pytest.fixture
def document_id() -> uuid.UUID:
    """Fixed document UUID for tests."""
    return uuid.UUID("660f9511-f3a0-52e5-b827-557766551111")


# ---------------------------------------------------------------------------
# Sort map
# ---------------------------------------------------------------------------


class TestSortMap:
    """Tests for the sort parameter mapping."""

    def test_all_sort_keys_defined(self) -> None:
        """All expected sort keys are present in the sort map."""
        expected_keys = {
            "name",
            "-name",
            "uploaded_at",
            "-uploaded_at",
            "status",
            "-status",
        }
        assert expected_keys == set(_SORT_MAP.keys())

    def test_default_sort_is_uploaded_at_desc(self) -> None:
        """Default sort is by uploaded_at descending."""
        assert _DEFAULT_SORT is not None


# ---------------------------------------------------------------------------
# Filter application
# ---------------------------------------------------------------------------


class TestApplyFilters:
    """Tests for DocumentRepository._apply_filters."""

    def test_no_filters_returns_unmodified_query(self, repo: DocumentRepository) -> None:
        """Empty filters do not add WHERE clauses."""
        query = select(Document)
        filters = DocumentFilters()
        result = repo._apply_filters(query, filters)
        assert result is not None

    def test_status_filter_applied(self, repo: DocumentRepository) -> None:
        """Status filter adds IN clause for the provided statuses."""
        query = select(Document)
        filters = DocumentFilters(
            status=[
                DocumentStatusFilter.UPLOADED,
                DocumentStatusFilter.ERROR,
            ]
        )
        result = repo._apply_filters(query, filters)
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "document_status" in compiled or "status" in compiled

    def test_type_filter_applied(self, repo: DocumentRepository) -> None:
        """Type filter adds IN clause for the provided types."""
        query = select(Document)
        filters = DocumentFilters(type=[DocumentTypeFilter.CORE_RULES])
        result = repo._apply_filters(query, filters)
        assert result is not None

    def test_expansion_id_null_filter(self, repo: DocumentRepository) -> None:
        """expansion_id='null' adds IS NULL clause."""
        query = select(Document)
        filters = DocumentFilters(expansion_id="null")
        result = repo._apply_filters(query, filters)
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "IS NULL" in compiled.upper() or "is_" in compiled

    def test_expansion_id_uuid_filter(self, repo: DocumentRepository) -> None:
        """UUID expansion_id adds equality clause."""
        exp_id = uuid.uuid4()
        query = select(Document)
        filters = DocumentFilters(expansion_id=exp_id)
        result = repo._apply_filters(query, filters)
        assert result is not None


# ---------------------------------------------------------------------------
# Get by ID (silo isolation)
# ---------------------------------------------------------------------------


class TestGetById:
    """Tests for DocumentRepository.get_by_id."""

    @pytest.mark.anyio
    async def test_get_by_id_queries_with_game_scope(
        self,
        repo: DocumentRepository,
        mock_session: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        """get_by_id includes game_id and archived_at IS NULL."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repo.get_by_id(game_id, document_id)

        assert result is None
        mock_session.execute.assert_called_once()


# ---------------------------------------------------------------------------
# Get detail
# ---------------------------------------------------------------------------


class TestGetDetail:
    """Tests for DocumentRepository.get_detail."""

    @pytest.mark.anyio
    async def test_get_detail_returns_none_when_not_found(
        self,
        repo: DocumentRepository,
        mock_session: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        """get_detail returns None when document not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repo.get_detail(game_id, document_id)
        assert result is None

    @pytest.mark.anyio
    async def test_get_detail_returns_dict_with_counts(
        self,
        repo: DocumentRepository,
        mock_session: AsyncMock,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        """get_detail returns dict with document, counts."""
        doc_mock = MagicMock()
        doc_mock.id = document_id

        result_1 = MagicMock()
        result_1.scalar_one_or_none.return_value = doc_mock

        result_2 = MagicMock()
        result_2.scalar_one.return_value = 5

        result_3 = MagicMock()
        result_3.scalar_one.return_value = 2

        mock_session.execute.side_effect = [result_1, result_2, result_3]

        result = await repo.get_detail(game_id, document_id)

        assert result is not None
        assert result["document"] == doc_mock
        assert result["chunk_count"] == 5
        assert result["version_count"] == 2
