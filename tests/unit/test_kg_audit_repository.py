"""Unit tests for KGAuditRepository.

Tests audit log entry creation, game-scoped listing with filters,
and event counting.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from tabletop_oracle.repositories.kg_audit_repository import KGAuditRepository

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session() -> AsyncMock:
    """Build a mock AsyncSession."""
    session = AsyncMock()
    session.add = MagicMock()
    return session


def _make_audit_entry(
    *,
    game_id=None,
    event_type="extraction_complete",
    document_id=None,
    concept_id=None,
    details=None,
):
    """Build a mock KGAuditLog entry."""
    entry = MagicMock()
    entry.id = uuid4()
    entry.game_id = game_id or uuid4()
    entry.event_type = event_type
    entry.document_id = document_id
    entry.concept_id = concept_id
    entry.details = details or {}
    return entry


# ---------------------------------------------------------------------------
# Tests: create
# ---------------------------------------------------------------------------


class TestCreate:
    """Tests for KGAuditRepository.create."""

    @pytest.mark.asyncio
    async def test_adds_entry_and_flushes(self) -> None:
        """create should add the entry to the session and flush."""
        session = _make_session()
        repo = KGAuditRepository(session)
        game_id = uuid4()
        doc_id = uuid4()

        with patch("tabletop_oracle.repositories.kg_audit_repository.KGAuditLog") as mock_cls:
            mock_entry = MagicMock()
            mock_cls.return_value = mock_entry

            result = await repo.create(
                game_id=game_id,
                event_type="extraction_complete",
                details={"concept_count": 5},
                document_id=doc_id,
            )

        session.add.assert_called_once_with(mock_entry)
        session.flush.assert_awaited_once()
        assert result is mock_entry

    @pytest.mark.asyncio
    async def test_with_concept_id_passes_to_model(self) -> None:
        """create should pass concept_id to the KGAuditLog constructor."""
        session = _make_session()
        repo = KGAuditRepository(session)
        concept_id = uuid4()

        with patch("tabletop_oracle.repositories.kg_audit_repository.KGAuditLog") as mock_cls:
            mock_cls.return_value = MagicMock()

            await repo.create(
                game_id=uuid4(),
                event_type="conflict_detected",
                details={},
                concept_id=concept_id,
            )

        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["concept_id"] == concept_id

    @pytest.mark.asyncio
    async def test_without_optionals_passes_none(self) -> None:
        """create without document_id or concept_id passes None."""
        session = _make_session()
        repo = KGAuditRepository(session)

        with patch("tabletop_oracle.repositories.kg_audit_repository.KGAuditLog") as mock_cls:
            mock_cls.return_value = MagicMock()

            await repo.create(
                game_id=uuid4(),
                event_type="graph_rebuilt",
                details={"document_count": 3},
            )

        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["document_id"] is None
        assert call_kwargs["concept_id"] is None


# ---------------------------------------------------------------------------
# Tests: list_by_game
# ---------------------------------------------------------------------------


class TestListByGame:
    """Tests for KGAuditRepository.list_by_game."""

    @pytest.mark.asyncio
    async def test_executes_query(self) -> None:
        """list_by_game should execute a select query and return results."""
        session = _make_session()
        repo = KGAuditRepository(session)
        game_id = uuid4()

        mock_entries = [_make_audit_entry(), _make_audit_entry()]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_entries
        session.execute.return_value = mock_result

        result = await repo.list_by_game(game_id)

        session.execute.assert_awaited_once()
        assert result == mock_entries

    @pytest.mark.asyncio
    async def test_empty_result_returns_empty_list(self) -> None:
        """list_by_game returns empty list when no entries exist."""
        session = _make_session()
        repo = KGAuditRepository(session)

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        session.execute.return_value = mock_result

        result = await repo.list_by_game(uuid4())

        assert result == []


# ---------------------------------------------------------------------------
# Tests: count_by_game
# ---------------------------------------------------------------------------


class TestCountByGame:
    """Tests for KGAuditRepository.count_by_game."""

    @pytest.mark.asyncio
    async def test_returns_count(self) -> None:
        """count_by_game should return the scalar count."""
        session = _make_session()
        repo = KGAuditRepository(session)

        mock_result = MagicMock()
        mock_result.scalar_one.return_value = 42
        session.execute.return_value = mock_result

        result = await repo.count_by_game(uuid4())

        assert result == 42

    @pytest.mark.asyncio
    async def test_zero_entries_returns_zero(self) -> None:
        """count_by_game returns 0 when no entries exist."""
        session = _make_session()
        repo = KGAuditRepository(session)

        mock_result = MagicMock()
        mock_result.scalar_one.return_value = 0
        session.execute.return_value = mock_result

        result = await repo.count_by_game(uuid4())

        assert result == 0
