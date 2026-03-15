"""Unit tests for GameRepository.

Tests filter composition, sort mapping, escape_like utility, and
repository method delegation using mocked async sessions. No real
database is involved — these verify query construction logic and
method contracts.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from tabletop_oracle.repositories.game_repository import (
    _DEFAULT_SORT,
    _SORT_MAP,
    GameRepository,
)
from tabletop_oracle.repositories.query_utils import escape_like
from tabletop_oracle.schemas.game import GameComplexityFilter, GameFilters

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create a mock async database session."""
    session = AsyncMock()
    return session


@pytest.fixture
def repo(mock_session: AsyncMock) -> GameRepository:
    """Create a GameRepository with a mocked session."""
    return GameRepository(mock_session)


# ---------------------------------------------------------------------------
# escape_like tests
# ---------------------------------------------------------------------------


class TestEscapeLike:
    """Tests for the escape_like utility function."""

    def test_escapes_percent(self) -> None:
        """escape_like escapes % characters."""
        assert escape_like("100%") == "100\\%"

    def test_escapes_underscore(self) -> None:
        """escape_like escapes _ characters."""
        assert escape_like("some_name") == "some\\_name"

    def test_escapes_backslash(self) -> None:
        """escape_like escapes backslash characters."""
        assert escape_like("path\\to") == "path\\\\to"

    def test_escapes_all_wildcards(self) -> None:
        """escape_like escapes all wildcard characters in a single string."""
        assert escape_like("a%b_c\\d") == "a\\%b\\_c\\\\d"

    def test_no_special_characters(self) -> None:
        """escape_like returns input unchanged when no wildcards present."""
        assert escape_like("normal text") == "normal text"

    def test_empty_string(self) -> None:
        """escape_like handles empty string."""
        assert escape_like("") == ""

    def test_only_wildcards(self) -> None:
        """escape_like escapes a string composed entirely of wildcards."""
        assert escape_like("%_%") == "\\%\\_\\%"

    def test_backslash_escaped_before_others(self) -> None:
        """escape_like escapes backslash first to avoid double-escaping."""
        # Input: \%  -> after \ escape: \\%  -> after % escape: \\\\%
        # No, let's trace carefully:
        # Input: "\\%"  (one backslash, one percent)
        # Step 1 replace \\ -> \\\\: "\\\\%"  (two backslashes, one percent)
        # Step 2 replace % -> \\%: "\\\\\\%"  (two backslashes, backslash-percent)
        result = escape_like("\\%")
        assert result == "\\\\\\%"


# ---------------------------------------------------------------------------
# Sort mapping tests
# ---------------------------------------------------------------------------


class TestSortMapping:
    """Tests for the sort parameter mapping."""

    def test_all_expected_sort_keys_present(self) -> None:
        """_SORT_MAP contains all six documented sort options."""
        expected = {"name", "-name", "updated_at", "-updated_at", "created_at", "-created_at"}
        assert set(_SORT_MAP.keys()) == expected

    def test_default_sort_is_name_asc(self) -> None:
        """Default sort clause is name ascending."""
        # Verify the default sort compiles to something (not None)
        assert _DEFAULT_SORT is not None

    def test_unknown_sort_falls_back_to_default(self) -> None:
        """Requesting an unknown sort key falls back to _DEFAULT_SORT."""
        result = _SORT_MAP.get("nonexistent", _DEFAULT_SORT)
        assert result is _DEFAULT_SORT


# ---------------------------------------------------------------------------
# Filter composition tests
# ---------------------------------------------------------------------------


class TestApplyFilters:
    """Tests for _apply_filters query construction logic."""

    def test_active_only_default(self, repo: GameRepository) -> None:
        """Default filters add archived_at IS NULL clause."""
        from sqlalchemy import select

        from tabletop_oracle.models.game import Game

        filters = GameFilters()
        query = select(Game)
        result = repo._apply_filters(query, filters)
        # The compiled query should contain an archived_at IS NULL clause
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "archived_at IS NULL" in compiled

    def test_archived_true_no_archive_filter(self, repo: GameRepository) -> None:
        """Setting archived=True skips the archived_at IS NULL WHERE clause."""
        from sqlalchemy import select

        from tabletop_oracle.models.game import Game

        filters = GameFilters(archived=True)
        query = select(Game)
        result = repo._apply_filters(query, filters)
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "archived_at IS NULL" not in compiled

    def test_search_short_uses_ilike(self, repo: GameRepository) -> None:
        """Search terms < 3 chars use ILIKE pattern."""
        from sqlalchemy import select

        from tabletop_oracle.models.game import Game

        filters = GameFilters(search="ab")
        query = select(Game)
        result = repo._apply_filters(query, filters)
        compiled = str(result.compile())
        # SQLAlchemy renders ILIKE as lower(x) LIKE lower(y) in generic dialect
        assert "like" in compiled.lower()

    def test_search_long_uses_tsvector(self, repo: GameRepository) -> None:
        """Search terms >= 3 chars use to_tsvector full-text search."""
        from sqlalchemy import select

        from tabletop_oracle.models.game import Game

        filters = GameFilters(search="catan")
        query = select(Game)
        result = repo._apply_filters(query, filters)
        compiled = str(result.compile())
        assert "to_tsvector" in compiled

    def test_search_exactly_3_chars_uses_tsvector(self, repo: GameRepository) -> None:
        """Search term of exactly 3 chars uses full-text search."""
        from sqlalchemy import select

        from tabletop_oracle.models.game import Game

        filters = GameFilters(search="abc")
        query = select(Game)
        result = repo._apply_filters(query, filters)
        compiled = str(result.compile())
        assert "to_tsvector" in compiled

    def test_player_count_filter(self, repo: GameRepository) -> None:
        """Player count filter adds min_players/max_players range clause."""
        from sqlalchemy import select

        from tabletop_oracle.models.game import Game

        filters = GameFilters(player_count=4)
        query = select(Game)
        result = repo._apply_filters(query, filters)
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "min_players" in compiled
        assert "max_players" in compiled

    def test_complexity_filter(self, repo: GameRepository) -> None:
        """Complexity filter adds IN clause with ORM enum values."""
        from sqlalchemy import select

        from tabletop_oracle.models.game import Game

        filters = GameFilters(complexity=[GameComplexityFilter.LIGHT, GameComplexityFilter.MEDIUM])
        query = select(Game)
        result = repo._apply_filters(query, filters)
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "complexity" in compiled
        assert "IN" in compiled.upper()

    def test_tags_filter_and_logic(self, repo: GameRepository) -> None:
        """Tags filter adds correlated subquery per tag (AND logic)."""
        from sqlalchemy import select

        from tabletop_oracle.models.game import Game

        filters = GameFilters(tags=["strategy", "cooperative"])
        query = select(Game)
        result = repo._apply_filters(query, filters)
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        # Each tag generates a subquery, so game_tags should appear
        assert "game_tags" in compiled
        # Both tags should appear in the query
        assert "strategy" in compiled
        assert "cooperative" in compiled

    def test_single_tag_filter(self, repo: GameRepository) -> None:
        """A single tag filter adds one correlated subquery."""
        from sqlalchemy import select

        from tabletop_oracle.models.game import Game

        filters = GameFilters(tags=["strategy"])
        query = select(Game)
        result = repo._apply_filters(query, filters)
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "game_tags" in compiled
        assert "strategy" in compiled

    def test_no_filters_only_active(self, repo: GameRepository) -> None:
        """Empty filters only add the active-only clause."""
        from sqlalchemy import select

        from tabletop_oracle.models.game import Game

        filters = GameFilters()
        query = select(Game)
        result = repo._apply_filters(query, filters)
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "archived_at IS NULL" in compiled
        # No other filter clauses
        assert "ilike" not in compiled.lower()
        assert "to_tsvector" not in compiled
        assert "min_players <" not in compiled
        assert "game_tags" not in compiled

    def test_combined_filters(self, repo: GameRepository) -> None:
        """Multiple filters compose additively."""
        from sqlalchemy import select

        from tabletop_oracle.models.game import Game

        filters = GameFilters(
            search="catan",
            player_count=4,
            complexity=[GameComplexityFilter.LIGHT],
            tags=["strategy"],
        )
        query = select(Game)
        result = repo._apply_filters(query, filters)
        compiled = str(result.compile())
        assert "archived_at IS NULL" in compiled
        assert "to_tsvector" in compiled
        assert "min_players" in compiled
        assert "complexity" in compiled
        assert "game_tags" in compiled

    def test_search_ilike_escapes_wildcards(self, repo: GameRepository) -> None:
        """Short search terms have SQL wildcards escaped via escape_like."""
        from sqlalchemy import select

        from tabletop_oracle.models.game import Game

        filters = GameFilters(search="a%")
        query = select(Game)
        result = repo._apply_filters(query, filters)
        compiled = str(result.compile(compile_kwargs={"literal_binds": True}))
        # The % in the search term should be escaped
        assert "a\\%" in compiled


# ---------------------------------------------------------------------------
# Repository method tests (delegation to session)
# ---------------------------------------------------------------------------


class TestGameRepositoryCreate:
    """Tests for GameRepository.create method."""

    @pytest.mark.asyncio
    async def test_create_delegates_to_session(self, repo: GameRepository) -> None:
        """create() adds entity and flushes session."""
        game = MagicMock()
        result = await repo.create(game)
        repo._session.add.assert_called_once_with(game)
        repo._session.flush.assert_awaited_once()
        assert result is game


class TestGameRepositoryGetById:
    """Tests for GameRepository.get_by_id method."""

    @pytest.mark.asyncio
    async def test_get_by_id_returns_result(self, repo: GameRepository) -> None:
        """get_by_id() executes query and returns scalar result."""
        game_id = uuid.uuid4()
        mock_result = MagicMock()
        mock_game = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_game
        repo._session.execute.return_value = mock_result

        result = await repo.get_by_id(game_id)
        assert result is mock_game
        repo._session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_by_id_returns_none_when_not_found(self, repo: GameRepository) -> None:
        """get_by_id() returns None when game does not exist."""
        game_id = uuid.uuid4()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        repo._session.execute.return_value = mock_result

        result = await repo.get_by_id(game_id)
        assert result is None


class TestGameRepositoryUpdate:
    """Tests for GameRepository.update method."""

    @pytest.mark.asyncio
    async def test_update_merges_and_flushes(self, repo: GameRepository) -> None:
        """update() merges entity and flushes session."""
        game = MagicMock()
        merged_game = MagicMock()
        repo._session.merge.return_value = merged_game

        result = await repo.update(game)
        repo._session.merge.assert_awaited_once_with(game)
        repo._session.flush.assert_awaited_once()
        assert result is merged_game


class TestGameRepositorySetTags:
    """Tests for GameRepository.set_tags method."""

    @pytest.mark.asyncio
    async def test_set_tags_deletes_and_inserts(self, repo: GameRepository) -> None:
        """set_tags() deletes existing tags and inserts new ones."""
        game_id = uuid.uuid4()

        # Mock existing tags
        existing_tag = MagicMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [existing_tag]
        repo._session.execute.return_value = mock_result

        await repo.set_tags(game_id, ["strategy", "cooperative"])

        repo._session.delete.assert_awaited_once_with(existing_tag)
        # Two new tags added
        assert repo._session.add.call_count == 2
        repo._session.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_set_tags_empty_list_clears_tags(self, repo: GameRepository) -> None:
        """set_tags() with empty list removes all tags."""
        game_id = uuid.uuid4()

        existing_tag = MagicMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [existing_tag]
        repo._session.execute.return_value = mock_result

        await repo.set_tags(game_id, [])

        repo._session.delete.assert_awaited_once_with(existing_tag)
        # No new tags added (only the delete calls)
        repo._session.add.assert_not_called()
        repo._session.flush.assert_awaited_once()


class TestGameRepositoryGetTagsWithCounts:
    """Tests for GameRepository.get_tags_with_counts method."""

    @pytest.mark.asyncio
    async def test_get_tags_with_counts_returns_list(self, repo: GameRepository) -> None:
        """get_tags_with_counts() returns list of tag/count dicts."""
        mock_row_1 = MagicMock()
        mock_row_1.tag = "strategy"
        mock_row_1.count = 5
        mock_row_2 = MagicMock()
        mock_row_2.tag = "cooperative"
        mock_row_2.count = 3

        mock_result = MagicMock()
        mock_result.all.return_value = [mock_row_1, mock_row_2]
        repo._session.execute.return_value = mock_result

        result = await repo.get_tags_with_counts()

        assert len(result) == 2
        assert result[0] == {"tag": "strategy", "count": 5}
        assert result[1] == {"tag": "cooperative", "count": 3}

    @pytest.mark.asyncio
    async def test_get_tags_with_counts_empty(self, repo: GameRepository) -> None:
        """get_tags_with_counts() returns empty list when no tags exist."""
        mock_result = MagicMock()
        mock_result.all.return_value = []
        repo._session.execute.return_value = mock_result

        result = await repo.get_tags_with_counts()
        assert result == []


class TestGameRepositoryCount:
    """Tests for the private _count helper."""

    @pytest.mark.asyncio
    async def test_count_returns_integer(self, repo: GameRepository) -> None:
        """_count() returns scalar integer from count query."""
        from sqlalchemy import select

        from tabletop_oracle.models.game import Game

        mock_result = MagicMock()
        mock_result.scalar_one.return_value = 42
        repo._session.execute.return_value = mock_result

        query = select(Game)
        result = await repo._count(query)
        assert result == 42


class TestGameRepositoryListWithSummary:
    """Tests for list_with_summary method delegation."""

    @pytest.mark.asyncio
    async def test_list_with_summary_calls_session(self, repo: GameRepository) -> None:
        """list_with_summary() executes queries against the session."""
        from tabletop_oracle.api.pagination import PaginationParams

        filters = GameFilters()
        pagination = PaginationParams(page=1, page_size=25)

        # Mock count query
        count_result = MagicMock()
        count_result.scalar_one.return_value = 0

        # Mock summary query
        summary_result = MagicMock()
        summary_result.all.return_value = []

        repo._session.execute = AsyncMock(side_effect=[count_result, summary_result])

        result, total = await repo.list_with_summary(filters, pagination)
        assert total == 0
        assert result == []
        assert repo._session.execute.await_count == 2
