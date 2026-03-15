"""Unit tests for KGSummaryService.

Tests aggregate metrics, document contribution queries, and reprocess
counting. Uses mocked AsyncSession to avoid database dependencies.
"""

from __future__ import annotations

from datetime import UTC
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from tabletop_oracle.services.kg.summary import KGSummaryService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_db() -> AsyncMock:
    """Create a mock AsyncSession with a chainable execute pattern."""
    db = AsyncMock()
    return db


def _mock_scalar_result(value: object) -> MagicMock:
    """Create a mock result whose scalar_one() returns the given value."""
    result = MagicMock()
    result.scalar_one.return_value = value
    return result


def _mock_scalar_one_or_none_result(value: object) -> MagicMock:
    """Create a mock result whose scalar_one_or_none() returns the given value."""
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    return result


def _mock_rows_result(rows: list[object]) -> MagicMock:
    """Create a mock result whose all() returns the given rows."""
    result = MagicMock()
    result.all.return_value = rows
    return result


def _mock_first_result(row: object | None) -> MagicMock:
    """Create a mock result whose first() returns the given row."""
    result = MagicMock()
    result.first.return_value = row
    return result


# ---------------------------------------------------------------------------
# Tests: get_summary
# ---------------------------------------------------------------------------


class TestGetSummary:
    """Tests for KGSummaryService.get_summary."""

    @pytest.mark.asyncio
    async def test_returns_zero_counts_for_empty_graph(self) -> None:
        """get_summary should return zero counts when no KG data exists."""
        db = _mock_db()
        game_id = uuid4()

        # Mock execute calls in order:
        # 1. concept count = 0
        # 2. association count = 0
        # 3. document count = 0
        # 4. semantic types = []
        # 5. document summaries subquery (concept contributions) = []
        # 6. last updated concept = None
        # 7. last updated association = None
        db.execute = AsyncMock(
            side_effect=[
                _mock_scalar_result(0),  # concept_count
                _mock_scalar_result(0),  # association_count
                _mock_scalar_result(0),  # document_count
                _mock_rows_result([]),  # semantic_types
                _mock_rows_result([]),  # document summaries
                _mock_scalar_one_or_none_result(None),  # concept max updated_at
                _mock_scalar_one_or_none_result(None),  # assoc max updated_at
            ]
        )

        service = KGSummaryService(db)
        result = await service.get_summary(game_id)

        assert result.game_id == game_id
        assert result.concept_count == 0
        assert result.association_count == 0
        assert result.document_count == 0
        assert result.semantic_types == []
        assert result.documents == []
        assert result.last_updated_at is None

    @pytest.mark.asyncio
    async def test_returns_populated_counts(self) -> None:
        """get_summary should return correct counts when KG data exists."""
        from datetime import datetime

        db = _mock_db()
        game_id = uuid4()
        doc_id = uuid4()
        now = datetime(2026, 3, 14, 12, 30, 0, tzinfo=UTC)

        # Semantic type row mock
        type_row = MagicMock()
        type_row.semantic_type = "game_mechanic"
        type_row.cnt = 42

        # Document summary row mock
        doc_row = MagicMock()
        doc_row.document_id = doc_id
        doc_row.document_name = "Test Rules"
        doc_row.document_type = "core_rules"

        doc_row.concept_count = 10

        db.execute = AsyncMock(
            side_effect=[
                _mock_scalar_result(50),  # concept_count
                _mock_scalar_result(100),  # association_count
                _mock_scalar_result(2),  # document_count
                _mock_rows_result([type_row]),  # semantic_types
                _mock_rows_result([doc_row]),  # document summaries concept query
                _mock_scalar_result(25),  # association count for doc
                _mock_scalar_one_or_none_result(now),  # concept max updated_at
                _mock_scalar_one_or_none_result(None),  # assoc max updated_at
            ]
        )

        service = KGSummaryService(db)
        result = await service.get_summary(game_id)

        assert result.concept_count == 50
        assert result.association_count == 100
        assert result.document_count == 2
        assert len(result.semantic_types) == 1
        assert result.semantic_types[0].semantic_type == "game_mechanic"
        assert result.semantic_types[0].count == 42
        assert len(result.documents) == 1
        assert result.documents[0].document_id == doc_id
        assert result.documents[0].concepts_contributed == 10
        assert result.documents[0].associations_contributed == 25
        assert result.last_updated_at == now


# ---------------------------------------------------------------------------
# Tests: get_document_contributions
# ---------------------------------------------------------------------------


class TestGetDocumentContributions:
    """Tests for KGSummaryService.get_document_contributions."""

    @pytest.mark.asyncio
    async def test_returns_empty_for_no_contributions(self) -> None:
        """get_document_contributions should return empty list and zero count."""
        db = _mock_db()
        game_id = uuid4()

        db.execute = AsyncMock(
            side_effect=[
                _mock_scalar_result(0),  # total_count
                _mock_rows_result([]),  # doc_ids
            ]
        )

        service = KGSummaryService(db)
        contributions, total = await service.get_document_contributions(game_id)

        assert contributions == []
        assert total == 0

    @pytest.mark.asyncio
    async def test_returns_paginated_contributions(self) -> None:
        """get_document_contributions should return data for contributing docs."""
        from datetime import datetime

        db = _mock_db()
        game_id = uuid4()
        doc_id = uuid4()
        now = datetime(2026, 3, 14, 12, 30, 0, tzinfo=UTC)

        doc_id_row = MagicMock()
        doc_id_row.document_id = doc_id

        # Metadata row
        meta_row = MagicMock()
        meta_row.document_name = "Test Rules"
        meta_row.document_type = "core_rules"
        meta_row.expansion_id = None

        db.execute = AsyncMock(
            side_effect=[
                _mock_scalar_result(1),  # total_count
                _mock_rows_result([doc_id_row]),  # doc_ids
                _mock_first_result(meta_row),  # metadata
                _mock_scalar_result(15),  # concept_count
                _mock_scalar_result(30),  # association_count
                _mock_scalar_result(2),  # conflict_count
                _mock_scalar_one_or_none_result(now),  # last_processed_at
            ]
        )

        service = KGSummaryService(db)
        contributions, total = await service.get_document_contributions(
            game_id, limit=20, offset=0
        )

        assert total == 1
        assert len(contributions) == 1
        assert contributions[0].document_id == doc_id
        assert contributions[0].document_name == "Test Rules"
        assert contributions[0].document_type == "core_rules"
        assert contributions[0].expansion_id is None
        assert contributions[0].concepts_contributed == 15
        assert contributions[0].associations_contributed == 30
        assert contributions[0].conflicts_detected == 2
        assert contributions[0].last_processed_at == now


# ---------------------------------------------------------------------------
# Tests: count_documents_for_reprocess
# ---------------------------------------------------------------------------


class TestCountDocumentsForReprocess:
    """Tests for KGSummaryService.count_documents_for_reprocess."""

    @pytest.mark.asyncio
    async def test_returns_zero_for_empty_graph(self) -> None:
        """count_documents_for_reprocess should return 0 when no KG data exists."""
        db = _mock_db()
        game_id = uuid4()

        db.execute = AsyncMock(return_value=_mock_scalar_result(0))

        service = KGSummaryService(db)
        result = await service.count_documents_for_reprocess(game_id)

        assert result == 0

    @pytest.mark.asyncio
    async def test_returns_document_count(self) -> None:
        """count_documents_for_reprocess should return correct count."""
        db = _mock_db()
        game_id = uuid4()

        db.execute = AsyncMock(return_value=_mock_scalar_result(5))

        service = KGSummaryService(db)
        result = await service.count_documents_for_reprocess(game_id)

        assert result == 5
