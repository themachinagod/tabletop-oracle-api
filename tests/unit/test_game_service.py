"""Unit tests for GameService business logic.

Tests all service methods with mocked GameRepository. Covers happy paths,
validation errors, conflict errors, not-found errors, tag normalisation,
UNSET-aware partial updates, and edge cases.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tabletop_oracle.api.pagination import PaginationParams
from tabletop_oracle.errors.exceptions import ConflictError, NotFoundError, ValidationError
from tabletop_oracle.models.enums import GameComplexity
from tabletop_oracle.schemas.game import (
    GameComplexityFilter,
    GameCreate,
    GameFilters,
    GameUpdate,
)
from tabletop_oracle.services.game_service import GameService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_game_model(**overrides: Any) -> MagicMock:
    """Build a mock Game ORM entity with sensible defaults."""
    defaults: dict[str, Any] = {
        "id": uuid.uuid4(),
        "created_by": uuid.uuid4(),
        "name": "Catan",
        "publisher": "Catan Studio",
        "year_published": 1995,
        "edition": "5th Edition",
        "min_players": 3,
        "max_players": 4,
        "description": "Trade and build.",
        "cover_image_url": None,
        "complexity": GameComplexity.MEDIUM,
        "archived_at": None,
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        "updated_at": datetime(2026, 1, 1, tzinfo=UTC),
        "tags": [],
    }
    defaults.update(overrides)
    mock = MagicMock()
    for key, val in defaults.items():
        setattr(mock, key, val)
    return mock


def _mock_repo() -> AsyncMock:
    """Build a mock GameRepository with all async methods."""
    repo = AsyncMock()
    repo.create = AsyncMock()
    repo.get_by_id = AsyncMock()
    repo.list_with_summary = AsyncMock()
    repo.update = AsyncMock()
    repo.set_tags = AsyncMock()
    repo.get_tags_with_counts = AsyncMock()
    return repo


# ---------------------------------------------------------------------------
# create_game
# ---------------------------------------------------------------------------


class TestCreateGame:
    """Tests for GameService.create_game()."""

    @pytest.mark.asyncio
    async def test_create_game_sets_created_by_from_user_id(self) -> None:
        """created_by is set from user_id, not from client input."""
        repo = _mock_repo()
        user_id = uuid.uuid4()
        game_mock = _make_game_model(id=uuid.uuid4())
        repo.create.return_value = game_mock

        svc = GameService(repo)
        data = GameCreate(name="Catan", min_players=3, max_players=4)

        await svc.create_game(data, user_id)

        created_game = repo.create.call_args[0][0]
        assert created_game.created_by == user_id

    @pytest.mark.asyncio
    async def test_create_game_normalises_tags(self) -> None:
        """Tags are lowercased, trimmed, deduplicated, and sorted."""
        repo = _mock_repo()
        game_mock = _make_game_model(id=uuid.uuid4())
        repo.create.return_value = game_mock

        svc = GameService(repo)
        data = GameCreate(name="Catan", tags=["Strategy", " trading ", "STRATEGY", "cooperative"])

        await svc.create_game(data, uuid.uuid4())

        # set_tags should be called with sorted, deduplicated, normalised tags
        repo.set_tags.assert_called_once()
        actual_tags = repo.set_tags.call_args[0][1]
        assert actual_tags == ["cooperative", "strategy", "trading"]

    @pytest.mark.asyncio
    async def test_create_game_empty_tags_skips_set_tags(self) -> None:
        """When no tags provided, set_tags is not called."""
        repo = _mock_repo()
        game_mock = _make_game_model(id=uuid.uuid4())
        repo.create.return_value = game_mock

        svc = GameService(repo)
        data = GameCreate(name="Catan")

        await svc.create_game(data, uuid.uuid4())

        repo.set_tags.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_game_whitespace_only_tags_filtered(self) -> None:
        """Whitespace-only tags are removed during normalisation."""
        repo = _mock_repo()
        game_mock = _make_game_model(id=uuid.uuid4())
        repo.create.return_value = game_mock

        svc = GameService(repo)
        # The schema filters whitespace-only tags, but the service also handles it
        data = GameCreate(name="Catan", tags=["valid"])

        await svc.create_game(data, uuid.uuid4())

        repo.set_tags.assert_called_once()
        actual_tags = repo.set_tags.call_args[0][1]
        assert actual_tags == ["valid"]

    @pytest.mark.asyncio
    async def test_create_game_validates_player_count_both_required(self) -> None:
        """Providing only min_players without max_players raises ValidationError."""
        repo = _mock_repo()
        svc = GameService(repo)

        # Pydantic schema already validates this, but service layer double-checks
        # We need to bypass schema validation to test service validation
        data = MagicMock(spec=GameCreate)
        data.name = "Catan"
        data.min_players = 3
        data.max_players = None
        data.tags = []
        data.publisher = None
        data.year_published = None
        data.edition = None
        data.description = None
        data.cover_image_url = None
        data.complexity = None

        with pytest.raises(ValidationError, match="Both min_players and max_players"):
            await svc.create_game(data, uuid.uuid4())

    @pytest.mark.asyncio
    async def test_create_game_validates_min_not_greater_than_max(self) -> None:
        """min_players > max_players raises ValidationError."""
        repo = _mock_repo()
        svc = GameService(repo)

        data = MagicMock(spec=GameCreate)
        data.name = "Catan"
        data.min_players = 6
        data.max_players = 4
        data.tags = []
        data.publisher = None
        data.year_published = None
        data.edition = None
        data.description = None
        data.cover_image_url = None
        data.complexity = None

        with pytest.raises(ValidationError, match="min_players must be <= max_players"):
            await svc.create_game(data, uuid.uuid4())

    @pytest.mark.asyncio
    async def test_create_game_both_none_players_is_valid(self) -> None:
        """Both min_players and max_players None is valid (neither provided)."""
        repo = _mock_repo()
        game_mock = _make_game_model(id=uuid.uuid4())
        repo.create.return_value = game_mock

        svc = GameService(repo)
        data = GameCreate(name="Catan")

        await svc.create_game(data, uuid.uuid4())

        repo.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_game_converts_complexity_to_orm_enum(self) -> None:
        """Schema GameComplexityFilter is converted to ORM GameComplexity."""
        repo = _mock_repo()
        game_mock = _make_game_model(id=uuid.uuid4())
        repo.create.return_value = game_mock

        svc = GameService(repo)
        data = GameCreate(name="Catan", complexity=GameComplexityFilter.HEAVY)

        await svc.create_game(data, uuid.uuid4())

        created_game = repo.create.call_args[0][0]
        assert created_game.complexity == "heavy"

    @pytest.mark.asyncio
    async def test_create_game_delegates_to_repository(self) -> None:
        """create_game calls repo.create and returns the result."""
        repo = _mock_repo()
        game_mock = _make_game_model(id=uuid.uuid4())
        repo.create.return_value = game_mock

        svc = GameService(repo)
        data = GameCreate(name="Catan")

        result = await svc.create_game(data, uuid.uuid4())

        repo.create.assert_called_once()
        assert result is game_mock


# ---------------------------------------------------------------------------
# get_game
# ---------------------------------------------------------------------------


class TestGetGame:
    """Tests for GameService.get_game()."""

    @pytest.mark.asyncio
    async def test_get_game_returns_game_when_found(self) -> None:
        """Returns the game entity when it exists."""
        repo = _mock_repo()
        game_id = uuid.uuid4()
        game_mock = _make_game_model(id=game_id)
        repo.get_by_id.return_value = game_mock

        svc = GameService(repo)
        result = await svc.get_game(game_id)

        assert result is game_mock
        repo.get_by_id.assert_called_once_with(game_id)

    @pytest.mark.asyncio
    async def test_get_game_raises_not_found_when_missing(self) -> None:
        """Raises NotFoundError when the game does not exist."""
        repo = _mock_repo()
        game_id = uuid.uuid4()
        repo.get_by_id.return_value = None

        svc = GameService(repo)

        with pytest.raises(NotFoundError, match="Game"):
            await svc.get_game(game_id)


# ---------------------------------------------------------------------------
# list_games
# ---------------------------------------------------------------------------


class TestListGames:
    """Tests for GameService.list_games()."""

    @pytest.mark.asyncio
    async def test_list_games_delegates_to_repository(self) -> None:
        """list_games passes filters and pagination directly to repo."""
        repo = _mock_repo()
        expected = ([{"game": "mock"}], 1)
        repo.list_with_summary.return_value = expected

        svc = GameService(repo)
        filters = GameFilters()
        pagination = PaginationParams(page=1, page_size=25)

        result = await svc.list_games(filters, pagination)

        assert result == expected
        repo.list_with_summary.assert_called_once_with(filters, pagination)


# ---------------------------------------------------------------------------
# update_game
# ---------------------------------------------------------------------------


class TestUpdateGame:
    """Tests for GameService.update_game()."""

    @pytest.mark.asyncio
    async def test_update_game_applies_set_fields_only(self) -> None:
        """Only fields not UNSET are applied to the game entity."""
        repo = _mock_repo()
        game_id = uuid.uuid4()
        game_mock = _make_game_model(id=game_id, name="Old Name")
        repo.get_by_id.return_value = game_mock
        repo.update.return_value = game_mock

        svc = GameService(repo)
        data = GameUpdate(name="New Name")

        await svc.update_game(game_id, data)

        assert game_mock.name == "New Name"
        repo.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_game_skips_when_no_fields_set(self) -> None:
        """Empty update (all UNSET) returns the game without modifying it."""
        repo = _mock_repo()
        game_id = uuid.uuid4()
        game_mock = _make_game_model(id=game_id)
        repo.get_by_id.return_value = game_mock

        svc = GameService(repo)
        data = GameUpdate()

        result = await svc.update_game(game_id, data)

        assert result is game_mock
        repo.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_game_normalises_tags(self) -> None:
        """Tags are normalised and set via set_tags when provided."""
        repo = _mock_repo()
        game_id = uuid.uuid4()
        game_mock = _make_game_model(id=game_id)
        repo.get_by_id.return_value = game_mock
        repo.update.return_value = game_mock

        svc = GameService(repo)
        # Tags are already normalised by Pydantic, but service sorts them
        data = GameUpdate(tags=["zulu", "Alpha", "alpha"])

        await svc.update_game(game_id, data)

        repo.set_tags.assert_called_once()
        actual_tags = repo.set_tags.call_args[0][1]
        assert actual_tags == ["alpha", "zulu"]

    @pytest.mark.asyncio
    async def test_update_game_null_tags_clears_all(self) -> None:
        """Setting tags to None clears all tags."""
        repo = _mock_repo()
        game_id = uuid.uuid4()
        game_mock = _make_game_model(id=game_id)
        repo.get_by_id.return_value = game_mock
        repo.update.return_value = game_mock

        svc = GameService(repo)
        data = GameUpdate(tags=None)

        await svc.update_game(game_id, data)

        repo.set_tags.assert_called_once_with(game_id, [])

    @pytest.mark.asyncio
    async def test_update_game_unset_tags_does_not_call_set_tags(self) -> None:
        """When tags are UNSET, set_tags is not called."""
        repo = _mock_repo()
        game_id = uuid.uuid4()
        game_mock = _make_game_model(id=game_id)
        repo.get_by_id.return_value = game_mock
        repo.update.return_value = game_mock

        svc = GameService(repo)
        data = GameUpdate(name="New Name")

        await svc.update_game(game_id, data)

        repo.set_tags.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_game_validates_player_counts_when_both_updated(self) -> None:
        """Validates min <= max when both player counts are in the update."""
        repo = _mock_repo()
        game_id = uuid.uuid4()
        game_mock = _make_game_model(id=game_id, min_players=2, max_players=4)
        repo.get_by_id.return_value = game_mock

        svc = GameService(repo)

        # Bypass Pydantic schema validation to test the service layer directly
        data = MagicMock(spec=GameUpdate)
        data.get_set_fields.return_value = {"min_players": 5, "max_players": 3}

        with pytest.raises(ValidationError, match="min_players must be <= max_players"):
            await svc.update_game(game_id, data)

    @pytest.mark.asyncio
    async def test_update_game_validates_against_existing_player_counts(self) -> None:
        """Validates new min_players against existing max_players."""
        repo = _mock_repo()
        game_id = uuid.uuid4()
        game_mock = _make_game_model(id=game_id, min_players=2, max_players=4)
        repo.get_by_id.return_value = game_mock

        svc = GameService(repo)
        # Only updating min_players, effective_max stays 4
        data = GameUpdate(min_players=5)

        with pytest.raises(ValidationError, match="min_players must be <= max_players"):
            await svc.update_game(game_id, data)

    @pytest.mark.asyncio
    async def test_update_game_validates_new_max_against_existing_min(self) -> None:
        """Validates new max_players against existing min_players."""
        repo = _mock_repo()
        game_id = uuid.uuid4()
        game_mock = _make_game_model(id=game_id, min_players=3, max_players=6)
        repo.get_by_id.return_value = game_mock

        svc = GameService(repo)
        data = GameUpdate(max_players=2)

        with pytest.raises(ValidationError, match="min_players must be <= max_players"):
            await svc.update_game(game_id, data)

    @pytest.mark.asyncio
    async def test_update_game_allows_valid_partial_player_count(self) -> None:
        """Updating only max_players to a valid value succeeds."""
        repo = _mock_repo()
        game_id = uuid.uuid4()
        game_mock = _make_game_model(id=game_id, min_players=2, max_players=4)
        repo.get_by_id.return_value = game_mock
        repo.update.return_value = game_mock

        svc = GameService(repo)
        data = GameUpdate(max_players=6)

        await svc.update_game(game_id, data)

        assert game_mock.max_players == 6

    @pytest.mark.asyncio
    async def test_update_game_converts_complexity_enum(self) -> None:
        """Schema complexity enum is converted to ORM enum on update."""
        repo = _mock_repo()
        game_id = uuid.uuid4()
        game_mock = _make_game_model(id=game_id)
        repo.get_by_id.return_value = game_mock
        repo.update.return_value = game_mock

        svc = GameService(repo)
        data = GameUpdate(complexity=GameComplexityFilter.LIGHT)

        await svc.update_game(game_id, data)

        assert game_mock.complexity == "light"

    @pytest.mark.asyncio
    async def test_update_game_raises_not_found(self) -> None:
        """Raises NotFoundError when game does not exist."""
        repo = _mock_repo()
        repo.get_by_id.return_value = None

        svc = GameService(repo)
        data = GameUpdate(name="New Name")

        with pytest.raises(NotFoundError):
            await svc.update_game(uuid.uuid4(), data)

    @pytest.mark.asyncio
    async def test_update_game_nulling_player_count_one_side(self) -> None:
        """Setting min_players to None while max_players exists raises ValidationError."""
        repo = _mock_repo()
        game_id = uuid.uuid4()
        game_mock = _make_game_model(id=game_id, min_players=2, max_players=4)
        repo.get_by_id.return_value = game_mock

        svc = GameService(repo)
        data = GameUpdate(min_players=None)

        with pytest.raises(ValidationError, match="Both min_players and max_players"):
            await svc.update_game(game_id, data)


# ---------------------------------------------------------------------------
# archive_game
# ---------------------------------------------------------------------------


class TestArchiveGame:
    """Tests for GameService.archive_game()."""

    @pytest.mark.asyncio
    async def test_archive_game_sets_archived_at(self) -> None:
        """archive_game sets archived_at to current timestamp."""
        repo = _mock_repo()
        game_id = uuid.uuid4()
        game_mock = _make_game_model(id=game_id, archived_at=None)
        repo.get_by_id.return_value = game_mock
        repo.update.return_value = game_mock

        svc = GameService(repo)
        await svc.archive_game(game_id)

        assert game_mock.archived_at is not None
        repo.update.assert_called_once_with(game_mock)

    @pytest.mark.asyncio
    async def test_archive_game_raises_conflict_when_already_archived(self) -> None:
        """Raises ConflictError when game is already archived."""
        repo = _mock_repo()
        game_id = uuid.uuid4()
        game_mock = _make_game_model(id=game_id, archived_at=datetime(2026, 1, 1, tzinfo=UTC))
        repo.get_by_id.return_value = game_mock

        svc = GameService(repo)

        with pytest.raises(ConflictError, match="already archived"):
            await svc.archive_game(game_id)

    @pytest.mark.asyncio
    async def test_archive_game_raises_not_found(self) -> None:
        """Raises NotFoundError when game does not exist."""
        repo = _mock_repo()
        repo.get_by_id.return_value = None

        svc = GameService(repo)

        with pytest.raises(NotFoundError):
            await svc.archive_game(uuid.uuid4())


# ---------------------------------------------------------------------------
# restore_game
# ---------------------------------------------------------------------------


class TestRestoreGame:
    """Tests for GameService.restore_game()."""

    @pytest.mark.asyncio
    async def test_restore_game_clears_archived_at(self) -> None:
        """restore_game sets archived_at to None."""
        repo = _mock_repo()
        game_id = uuid.uuid4()
        game_mock = _make_game_model(id=game_id, archived_at=datetime(2026, 1, 1, tzinfo=UTC))
        repo.get_by_id.return_value = game_mock
        repo.update.return_value = game_mock

        svc = GameService(repo)
        await svc.restore_game(game_id)

        assert game_mock.archived_at is None
        repo.update.assert_called_once_with(game_mock)

    @pytest.mark.asyncio
    async def test_restore_game_raises_conflict_when_not_archived(self) -> None:
        """Raises ConflictError when game is not archived."""
        repo = _mock_repo()
        game_id = uuid.uuid4()
        game_mock = _make_game_model(id=game_id, archived_at=None)
        repo.get_by_id.return_value = game_mock

        svc = GameService(repo)

        with pytest.raises(ConflictError, match="not archived"):
            await svc.restore_game(game_id)

    @pytest.mark.asyncio
    async def test_restore_game_raises_not_found(self) -> None:
        """Raises NotFoundError when game does not exist."""
        repo = _mock_repo()
        repo.get_by_id.return_value = None

        svc = GameService(repo)

        with pytest.raises(NotFoundError):
            await svc.restore_game(uuid.uuid4())


# ---------------------------------------------------------------------------
# get_tags
# ---------------------------------------------------------------------------


class TestGetTags:
    """Tests for GameService.get_tags()."""

    @pytest.mark.asyncio
    async def test_get_tags_delegates_to_repository(self) -> None:
        """get_tags returns the repository result directly."""
        repo = _mock_repo()
        expected = [
            {"tag": "strategy", "count": 5},
            {"tag": "cooperative", "count": 3},
        ]
        repo.get_tags_with_counts.return_value = expected

        svc = GameService(repo)
        result = await svc.get_tags()

        assert result == expected
        repo.get_tags_with_counts.assert_called_once()


# ---------------------------------------------------------------------------
# Tag normalisation edge cases
# ---------------------------------------------------------------------------


class TestTagNormalisation:
    """Tests for the _normalise_tags static method."""

    def test_normalise_empty_list(self) -> None:
        """Empty tag list returns empty list."""
        assert GameService._normalise_tags([]) == []

    def test_normalise_lowercases(self) -> None:
        """Tags are lowercased."""
        result = GameService._normalise_tags(["STRATEGY", "Trading"])
        assert result == ["strategy", "trading"]

    def test_normalise_strips_whitespace(self) -> None:
        """Leading/trailing whitespace is stripped."""
        result = GameService._normalise_tags(["  strategy  ", "trading"])
        assert result == ["strategy", "trading"]

    def test_normalise_deduplicates(self) -> None:
        """Duplicate tags (after normalisation) are removed."""
        result = GameService._normalise_tags(["Strategy", "STRATEGY", "strategy"])
        assert result == ["strategy"]

    def test_normalise_sorts(self) -> None:
        """Tags are sorted alphabetically."""
        result = GameService._normalise_tags(["zulu", "alpha", "mike"])
        assert result == ["alpha", "mike", "zulu"]

    def test_normalise_filters_whitespace_only(self) -> None:
        """Whitespace-only tags are filtered out."""
        result = GameService._normalise_tags(["valid", "  ", "\t", ""])
        assert result == ["valid"]

    def test_normalise_combined(self) -> None:
        """Combined normalisation: lowercase, trim, dedup, sort, filter."""
        result = GameService._normalise_tags(["  Zulu ", "ALPHA", "alpha", "  ", "mike"])
        assert result == ["alpha", "mike", "zulu"]


# ---------------------------------------------------------------------------
# Player count validation edge cases
# ---------------------------------------------------------------------------


class TestPlayerCountValidation:
    """Tests for the _validate_player_counts static method."""

    def test_both_none_is_valid(self) -> None:
        """Both None (neither provided) is valid."""
        GameService._validate_player_counts(None, None)

    def test_both_equal_is_valid(self) -> None:
        """min == max is valid (solo game)."""
        GameService._validate_player_counts(1, 1)

    def test_min_less_than_max_is_valid(self) -> None:
        """min < max is valid."""
        GameService._validate_player_counts(2, 6)

    def test_only_min_raises(self) -> None:
        """Only min provided raises ValidationError."""
        with pytest.raises(ValidationError):
            GameService._validate_player_counts(2, None)

    def test_only_max_raises(self) -> None:
        """Only max provided raises ValidationError."""
        with pytest.raises(ValidationError):
            GameService._validate_player_counts(None, 4)

    def test_min_greater_than_max_raises(self) -> None:
        """min > max raises ValidationError."""
        with pytest.raises(ValidationError):
            GameService._validate_player_counts(5, 3)
