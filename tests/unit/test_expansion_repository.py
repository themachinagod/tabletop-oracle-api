"""Unit tests for ExpansionRepository.

Tests filter composition, game silo isolation, method delegation, and
query construction using mocked async sessions. No real database is
involved — these verify query construction logic and method contracts.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from tabletop_oracle.api.pagination import PaginationParams
from tabletop_oracle.repositories.expansion_repository import ExpansionRepository
from tabletop_oracle.schemas.expansion import ExpansionFilters

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create a mock async database session."""
    return AsyncMock()


@pytest.fixture
def repo(mock_session: AsyncMock) -> ExpansionRepository:
    """Create an ExpansionRepository with a mocked session."""
    return ExpansionRepository(mock_session)


# ---------------------------------------------------------------------------
# Filter composition tests
# ---------------------------------------------------------------------------


class TestApplyFilters:
    """Tests for _apply_filters query construction logic."""

    def test_active_only_default(self, repo: ExpansionRepository) -> None:
        """Default filters add archived_at IS NULL clause."""
        from sqlalchemy import select

        from tabletop_oracle.models.expansion import Expansion

        filters = ExpansionFilters()
        query = select(Expansion)
        result = repo._apply_filters(query, filters)
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "archived_at IS NULL" in compiled

    def test_archived_true_no_archive_filter(self, repo: ExpansionRepository) -> None:
        """Setting archived=True skips the archived_at IS NULL WHERE clause."""
        from sqlalchemy import select

        from tabletop_oracle.models.expansion import Expansion

        filters = ExpansionFilters(archived=True)
        query = select(Expansion)
        result = repo._apply_filters(query, filters)
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "archived_at IS NULL" not in compiled


# ---------------------------------------------------------------------------
# Game silo isolation tests
# ---------------------------------------------------------------------------


class TestGameSiloIsolation:
    """Tests verifying game_id scoping as a security boundary."""

    @pytest.mark.asyncio
    async def test_get_by_id_returns_none_for_wrong_game(
        self,
        repo: ExpansionRepository,
    ) -> None:
        """get_by_id returns None when expansion belongs to a different game."""
        game_id = uuid.uuid4()
        expansion_id = uuid.uuid4()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        repo._session.execute.return_value = mock_result

        result = await repo.get_by_id(game_id, expansion_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_id_query_contains_game_id(
        self,
        repo: ExpansionRepository,
    ) -> None:
        """get_by_id constructs a query scoped to game_id."""
        game_id = uuid.uuid4()
        expansion_id = uuid.uuid4()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        repo._session.execute.return_value = mock_result

        await repo.get_by_id(game_id, expansion_id)

        # Inspect the statement passed to execute
        call_args = repo._session.execute.call_args
        stmt = call_args[0][0]
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        assert "game_id" in compiled
        # UUIDs may render with or without hyphens in compiled SQL
        assert game_id.hex in compiled or str(game_id) in compiled
        assert expansion_id.hex in compiled or str(expansion_id) in compiled

    @pytest.mark.asyncio
    async def test_list_by_game_scopes_to_game_id(
        self,
        repo: ExpansionRepository,
    ) -> None:
        """list_by_game only returns expansions for the specified game."""
        game_id = uuid.uuid4()
        filters = ExpansionFilters()
        pagination = PaginationParams(page=1, page_size=25)

        # Mock count query
        count_result = MagicMock()
        count_result.scalar_one.return_value = 0

        # Mock summary query
        summary_result = MagicMock()
        summary_result.all.return_value = []

        repo._session.execute = AsyncMock(side_effect=[count_result, summary_result])

        result, total = await repo.list_by_game(game_id, filters, pagination)
        assert total == 0
        assert result == []

        # Both queries should have been called
        assert repo._session.execute.await_count == 2


# ---------------------------------------------------------------------------
# Repository method tests (delegation to session)
# ---------------------------------------------------------------------------


class TestExpansionRepositoryCreate:
    """Tests for ExpansionRepository.create method (inherited from BaseRepository)."""

    @pytest.mark.asyncio
    async def test_create_delegates_to_session(self, repo: ExpansionRepository) -> None:
        """create() adds entity and flushes session."""
        expansion = MagicMock()
        result = await repo.create(expansion)
        repo._session.add.assert_called_once_with(expansion)
        repo._session.flush.assert_awaited_once()
        assert result is expansion


class TestExpansionRepositoryGetById:
    """Tests for ExpansionRepository.get_by_id method."""

    @pytest.mark.asyncio
    async def test_get_by_id_returns_result(self, repo: ExpansionRepository) -> None:
        """get_by_id() executes query and returns scalar result."""
        game_id = uuid.uuid4()
        expansion_id = uuid.uuid4()
        mock_result = MagicMock()
        mock_expansion = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_expansion
        repo._session.execute.return_value = mock_result

        result = await repo.get_by_id(game_id, expansion_id)
        assert result is mock_expansion
        repo._session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_by_id_returns_none_when_not_found(
        self,
        repo: ExpansionRepository,
    ) -> None:
        """get_by_id() returns None when expansion does not exist."""
        game_id = uuid.uuid4()
        expansion_id = uuid.uuid4()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        repo._session.execute.return_value = mock_result

        result = await repo.get_by_id(game_id, expansion_id)
        assert result is None


class TestExpansionRepositoryUpdate:
    """Tests for ExpansionRepository.update method."""

    @pytest.mark.asyncio
    async def test_update_merges_and_flushes(self, repo: ExpansionRepository) -> None:
        """update() merges entity and flushes session."""
        expansion = MagicMock()
        merged_expansion = MagicMock()
        repo._session.merge.return_value = merged_expansion

        result = await repo.update(expansion)
        repo._session.merge.assert_awaited_once_with(expansion)
        repo._session.flush.assert_awaited_once()
        assert result is merged_expansion


class TestExpansionRepositoryCount:
    """Tests for the private _count helper."""

    @pytest.mark.asyncio
    async def test_count_returns_integer(self, repo: ExpansionRepository) -> None:
        """_count() returns scalar integer from count query."""
        from sqlalchemy import select

        from tabletop_oracle.models.expansion import Expansion

        mock_result = MagicMock()
        mock_result.scalar_one.return_value = 7
        repo._session.execute.return_value = mock_result

        query = select(Expansion)
        result = await repo._count(query)
        assert result == 7


class TestExpansionRepositoryListByGame:
    """Tests for list_by_game method delegation and query construction."""

    @pytest.mark.asyncio
    async def test_list_by_game_calls_session(self, repo: ExpansionRepository) -> None:
        """list_by_game() executes two queries against the session."""
        game_id = uuid.uuid4()
        filters = ExpansionFilters()
        pagination = PaginationParams(page=1, page_size=25)

        count_result = MagicMock()
        count_result.scalar_one.return_value = 0

        summary_result = MagicMock()
        summary_result.all.return_value = []

        repo._session.execute = AsyncMock(side_effect=[count_result, summary_result])

        result, total = await repo.list_by_game(game_id, filters, pagination)
        assert total == 0
        assert result == []
        assert repo._session.execute.await_count == 2

    @pytest.mark.asyncio
    async def test_list_by_game_returns_expansion_dicts(
        self,
        repo: ExpansionRepository,
    ) -> None:
        """list_by_game() returns dicts with expansion and document_count."""
        game_id = uuid.uuid4()
        filters = ExpansionFilters()
        pagination = PaginationParams(page=1, page_size=25)

        count_result = MagicMock()
        count_result.scalar_one.return_value = 1

        mock_expansion = MagicMock()
        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self, idx: mock_expansion if idx == 0 else 3
        summary_result = MagicMock()
        summary_result.all.return_value = [mock_row]

        repo._session.execute = AsyncMock(
            side_effect=[count_result, summary_result],
        )

        result, total = await repo.list_by_game(game_id, filters, pagination)
        assert total == 1
        assert len(result) == 1
        assert result[0]["expansion"] is mock_expansion
        assert result[0]["document_count"] == 3

    @pytest.mark.asyncio
    async def test_list_by_game_with_archived_filter(
        self,
        repo: ExpansionRepository,
    ) -> None:
        """list_by_game() with archived=True does not filter by archived_at."""
        game_id = uuid.uuid4()
        filters = ExpansionFilters(archived=True)
        pagination = PaginationParams(page=1, page_size=25)

        count_result = MagicMock()
        count_result.scalar_one.return_value = 0

        summary_result = MagicMock()
        summary_result.all.return_value = []

        repo._session.execute = AsyncMock(side_effect=[count_result, summary_result])

        result, total = await repo.list_by_game(game_id, filters, pagination)
        assert total == 0
        assert result == []

    @pytest.mark.asyncio
    async def test_list_by_game_pagination(self, repo: ExpansionRepository) -> None:
        """list_by_game() respects pagination parameters."""
        game_id = uuid.uuid4()
        filters = ExpansionFilters()
        pagination = PaginationParams(page=2, page_size=10)

        count_result = MagicMock()
        count_result.scalar_one.return_value = 15

        summary_result = MagicMock()
        summary_result.all.return_value = []

        repo._session.execute = AsyncMock(side_effect=[count_result, summary_result])

        _, total = await repo.list_by_game(game_id, filters, pagination)
        assert total == 15


# ---------------------------------------------------------------------------
# ExpansionFilters schema tests
# ---------------------------------------------------------------------------


class TestExpansionFilters:
    """Tests for the ExpansionFilters Pydantic schema."""

    def test_default_archived_false(self) -> None:
        """Default archived filter is False (active only)."""
        filters = ExpansionFilters()
        assert filters.archived is False

    def test_archived_true(self) -> None:
        """Archived filter can be set to True."""
        filters = ExpansionFilters(archived=True)
        assert filters.archived is True
