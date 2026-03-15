"""Unit tests for ExpansionService business logic.

Tests all service methods with mocked repositories. Covers happy paths,
parent game validation, game silo enforcement, UNSET-aware partial updates,
archive/restore conflict handling, and edge cases.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from tabletop_oracle.errors.exceptions import ConflictError, NotFoundError
from tabletop_oracle.schemas.expansion import (
    ExpansionCreate,
    ExpansionFilters,
    ExpansionUpdate,
)
from tabletop_oracle.services.expansion_service import ExpansionService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_expansion_repo() -> AsyncMock:
    """Create a mock ExpansionRepository."""
    repo = AsyncMock()
    return repo


@pytest.fixture
def mock_game_repo() -> AsyncMock:
    """Create a mock GameRepository."""
    repo = AsyncMock()
    return repo


@pytest.fixture
def service(mock_expansion_repo: AsyncMock, mock_game_repo: AsyncMock) -> ExpansionService:
    """Create an ExpansionService with mocked repositories."""
    return ExpansionService(
        expansion_repo=mock_expansion_repo,
        game_repo=mock_game_repo,
    )


@pytest.fixture
def game_id() -> uuid.UUID:
    """Fixed game UUID for tests."""
    return uuid.UUID("550e8400-e29b-41d4-a716-446655440000")


@pytest.fixture
def expansion_id() -> uuid.UUID:
    """Fixed expansion UUID for tests."""
    return uuid.UUID("660f9511-f3a0-52e5-b827-557766551111")


@pytest.fixture
def user_id() -> uuid.UUID:
    """Fixed user UUID for tests."""
    return uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


def _make_expansion(
    *,
    expansion_id: uuid.UUID | None = None,
    game_id: uuid.UUID | None = None,
    name: str = "Seafarers of Catan",
    year_published: int | None = 1997,
    description: str | None = "Explore new islands.",
    archived_at: datetime | None = None,
) -> MagicMock:
    """Create a mock Expansion entity with specified attributes."""
    expansion = MagicMock()
    expansion.id = expansion_id or uuid.uuid4()
    expansion.game_id = game_id or uuid.uuid4()
    expansion.name = name
    expansion.year_published = year_published
    expansion.description = description
    expansion.archived_at = archived_at
    expansion.created_at = datetime(2026, 3, 14, 10, 0, 0, tzinfo=UTC)
    expansion.updated_at = datetime(2026, 3, 14, 10, 0, 0, tzinfo=UTC)
    return expansion


def _make_game(*, game_id: uuid.UUID | None = None) -> MagicMock:
    """Create a mock Game entity."""
    game = MagicMock()
    game.id = game_id or uuid.uuid4()
    game.archived_at = None
    return game


# ---------------------------------------------------------------------------
# create_expansion tests
# ---------------------------------------------------------------------------


class TestCreateExpansion:
    """Tests for ExpansionService.create_expansion."""

    @pytest.mark.asyncio
    async def test_create_expansion_happy_path(
        self,
        service: ExpansionService,
        mock_expansion_repo: AsyncMock,
        mock_game_repo: AsyncMock,
        game_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        """Successfully creates expansion when parent game exists."""
        mock_game_repo.get_by_id.return_value = _make_game(game_id=game_id)
        expected = _make_expansion(game_id=game_id)
        mock_expansion_repo.create.return_value = expected

        data = ExpansionCreate(
            name="Seafarers of Catan",
            year_published=1997,
            description="Explore new islands.",
        )

        result = await service.create_expansion(game_id, data, user_id)

        mock_game_repo.get_by_id.assert_awaited_once_with(game_id)
        mock_expansion_repo.create.assert_awaited_once()
        created_entity = mock_expansion_repo.create.call_args[0][0]
        assert created_entity.game_id == game_id
        assert created_entity.name == "Seafarers of Catan"
        assert created_entity.year_published == 1997
        assert created_entity.description == "Explore new islands."
        assert result is expected

    @pytest.mark.asyncio
    async def test_create_expansion_parent_game_not_found(
        self,
        service: ExpansionService,
        mock_game_repo: AsyncMock,
        game_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        """Raises NotFoundError when parent game does not exist."""
        mock_game_repo.get_by_id.return_value = None

        data = ExpansionCreate(name="Seafarers of Catan")

        with pytest.raises(NotFoundError, match="Game"):
            await service.create_expansion(game_id, data, user_id)

    @pytest.mark.asyncio
    async def test_create_expansion_minimal_fields(
        self,
        service: ExpansionService,
        mock_expansion_repo: AsyncMock,
        mock_game_repo: AsyncMock,
        game_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        """Creates expansion with only required fields (name)."""
        mock_game_repo.get_by_id.return_value = _make_game(game_id=game_id)
        expected = _make_expansion(game_id=game_id, year_published=None, description=None)
        mock_expansion_repo.create.return_value = expected

        data = ExpansionCreate(name="Minimal Expansion")

        await service.create_expansion(game_id, data, user_id)

        created_entity = mock_expansion_repo.create.call_args[0][0]
        assert created_entity.name == "Minimal Expansion"
        assert created_entity.year_published is None
        assert created_entity.description is None


# ---------------------------------------------------------------------------
# get_expansion tests
# ---------------------------------------------------------------------------


class TestGetExpansion:
    """Tests for ExpansionService.get_expansion."""

    @pytest.mark.asyncio
    async def test_get_expansion_happy_path(
        self,
        service: ExpansionService,
        mock_expansion_repo: AsyncMock,
        game_id: uuid.UUID,
        expansion_id: uuid.UUID,
    ) -> None:
        """Returns expansion when found and scoped to game."""
        expected = _make_expansion(expansion_id=expansion_id, game_id=game_id)
        mock_expansion_repo.get_by_id.return_value = expected

        result = await service.get_expansion(game_id, expansion_id)

        mock_expansion_repo.get_by_id.assert_awaited_once_with(game_id, expansion_id)
        assert result is expected

    @pytest.mark.asyncio
    async def test_get_expansion_not_found(
        self,
        service: ExpansionService,
        mock_expansion_repo: AsyncMock,
        game_id: uuid.UUID,
        expansion_id: uuid.UUID,
    ) -> None:
        """Raises NotFoundError when expansion not found."""
        mock_expansion_repo.get_by_id.return_value = None

        with pytest.raises(NotFoundError, match="Expansion"):
            await service.get_expansion(game_id, expansion_id)

    @pytest.mark.asyncio
    async def test_get_expansion_enforces_game_scoping(
        self,
        service: ExpansionService,
        mock_expansion_repo: AsyncMock,
        game_id: uuid.UUID,
        expansion_id: uuid.UUID,
    ) -> None:
        """Passes game_id to repo for silo enforcement."""
        mock_expansion_repo.get_by_id.return_value = None
        wrong_game_id = uuid.uuid4()

        with pytest.raises(NotFoundError):
            await service.get_expansion(wrong_game_id, expansion_id)

        mock_expansion_repo.get_by_id.assert_awaited_once_with(wrong_game_id, expansion_id)


# ---------------------------------------------------------------------------
# list_expansions tests
# ---------------------------------------------------------------------------


class TestListExpansions:
    """Tests for ExpansionService.list_expansions."""

    @pytest.mark.asyncio
    async def test_list_expansions_happy_path(
        self,
        service: ExpansionService,
        mock_expansion_repo: AsyncMock,
        mock_game_repo: AsyncMock,
        game_id: uuid.UUID,
    ) -> None:
        """Lists expansions when parent game exists."""
        from tabletop_oracle.api.pagination import PaginationParams

        mock_game_repo.get_by_id.return_value = _make_game(game_id=game_id)
        expected_result = ([{"expansion": _make_expansion(), "document_count": 3}], 1)
        mock_expansion_repo.list_by_game.return_value = expected_result

        filters = ExpansionFilters()
        pagination = PaginationParams(page=1, page_size=25)

        result = await service.list_expansions(game_id, filters, pagination)

        mock_game_repo.get_by_id.assert_awaited_once_with(game_id)
        mock_expansion_repo.list_by_game.assert_awaited_once_with(game_id, filters, pagination)
        assert result is expected_result

    @pytest.mark.asyncio
    async def test_list_expansions_parent_game_not_found(
        self,
        service: ExpansionService,
        mock_game_repo: AsyncMock,
        game_id: uuid.UUID,
    ) -> None:
        """Raises NotFoundError when parent game does not exist."""
        from tabletop_oracle.api.pagination import PaginationParams

        mock_game_repo.get_by_id.return_value = None

        filters = ExpansionFilters()
        pagination = PaginationParams(page=1, page_size=25)

        with pytest.raises(NotFoundError, match="Game"):
            await service.list_expansions(game_id, filters, pagination)

    @pytest.mark.asyncio
    async def test_list_expansions_with_archived_filter(
        self,
        service: ExpansionService,
        mock_expansion_repo: AsyncMock,
        mock_game_repo: AsyncMock,
        game_id: uuid.UUID,
    ) -> None:
        """Passes archived filter through to repository."""
        from tabletop_oracle.api.pagination import PaginationParams

        mock_game_repo.get_by_id.return_value = _make_game(game_id=game_id)
        mock_expansion_repo.list_by_game.return_value = ([], 0)

        filters = ExpansionFilters(archived=True)
        pagination = PaginationParams(page=1, page_size=25)

        await service.list_expansions(game_id, filters, pagination)

        call_filters = mock_expansion_repo.list_by_game.call_args[0][1]
        assert call_filters.archived is True


# ---------------------------------------------------------------------------
# update_expansion tests
# ---------------------------------------------------------------------------


class TestUpdateExpansion:
    """Tests for ExpansionService.update_expansion."""

    @pytest.mark.asyncio
    async def test_update_expansion_applies_set_fields(
        self,
        service: ExpansionService,
        mock_expansion_repo: AsyncMock,
        game_id: uuid.UUID,
        expansion_id: uuid.UUID,
    ) -> None:
        """Applies only non-UNSET fields to the expansion."""
        existing = _make_expansion(
            expansion_id=expansion_id,
            game_id=game_id,
            name="Old Name",
            description="Old desc",
        )
        mock_expansion_repo.get_by_id.return_value = existing
        mock_expansion_repo.update.return_value = existing

        data = ExpansionUpdate(name="New Name")

        await service.update_expansion(game_id, expansion_id, data)

        assert existing.name == "New Name"
        mock_expansion_repo.update.assert_awaited_once_with(existing)

    @pytest.mark.asyncio
    async def test_update_expansion_no_set_fields_skips_update(
        self,
        service: ExpansionService,
        mock_expansion_repo: AsyncMock,
        game_id: uuid.UUID,
        expansion_id: uuid.UUID,
    ) -> None:
        """Returns expansion unchanged when no fields are set."""
        existing = _make_expansion(expansion_id=expansion_id, game_id=game_id)
        mock_expansion_repo.get_by_id.return_value = existing

        data = ExpansionUpdate()  # All UNSET

        result = await service.update_expansion(game_id, expansion_id, data)

        assert result is existing
        mock_expansion_repo.update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_update_expansion_sets_field_to_null(
        self,
        service: ExpansionService,
        mock_expansion_repo: AsyncMock,
        game_id: uuid.UUID,
        expansion_id: uuid.UUID,
    ) -> None:
        """Explicitly setting a field to None applies the null value."""
        existing = _make_expansion(
            expansion_id=expansion_id,
            game_id=game_id,
            year_published=1997,
        )
        mock_expansion_repo.get_by_id.return_value = existing
        mock_expansion_repo.update.return_value = existing

        data = ExpansionUpdate(year_published=None)

        await service.update_expansion(game_id, expansion_id, data)

        assert existing.year_published is None
        mock_expansion_repo.update.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_update_expansion_not_found(
        self,
        service: ExpansionService,
        mock_expansion_repo: AsyncMock,
        game_id: uuid.UUID,
        expansion_id: uuid.UUID,
    ) -> None:
        """Raises NotFoundError when expansion does not exist."""
        mock_expansion_repo.get_by_id.return_value = None

        data = ExpansionUpdate(name="New Name")

        with pytest.raises(NotFoundError, match="Expansion"):
            await service.update_expansion(game_id, expansion_id, data)

    @pytest.mark.asyncio
    async def test_update_expansion_multiple_fields(
        self,
        service: ExpansionService,
        mock_expansion_repo: AsyncMock,
        game_id: uuid.UUID,
        expansion_id: uuid.UUID,
    ) -> None:
        """Applies multiple set fields in a single update."""
        existing = _make_expansion(expansion_id=expansion_id, game_id=game_id)
        mock_expansion_repo.get_by_id.return_value = existing
        mock_expansion_repo.update.return_value = existing

        data = ExpansionUpdate(
            name="Updated Name",
            year_published=2020,
            description="Updated description.",
        )

        await service.update_expansion(game_id, expansion_id, data)

        assert existing.name == "Updated Name"
        assert existing.year_published == 2020
        assert existing.description == "Updated description."


# ---------------------------------------------------------------------------
# archive_expansion tests
# ---------------------------------------------------------------------------


class TestArchiveExpansion:
    """Tests for ExpansionService.archive_expansion."""

    @pytest.mark.asyncio
    async def test_archive_expansion_happy_path(
        self,
        service: ExpansionService,
        mock_expansion_repo: AsyncMock,
        game_id: uuid.UUID,
        expansion_id: uuid.UUID,
    ) -> None:
        """Sets archived_at on an active expansion."""
        existing = _make_expansion(
            expansion_id=expansion_id,
            game_id=game_id,
            archived_at=None,
        )
        mock_expansion_repo.get_by_id.return_value = existing
        mock_expansion_repo.update.return_value = existing

        result = await service.archive_expansion(game_id, expansion_id)

        assert existing.archived_at is not None
        mock_expansion_repo.update.assert_awaited_once_with(existing)
        assert result is existing

    @pytest.mark.asyncio
    async def test_archive_expansion_already_archived(
        self,
        service: ExpansionService,
        mock_expansion_repo: AsyncMock,
        game_id: uuid.UUID,
        expansion_id: uuid.UUID,
    ) -> None:
        """Raises ConflictError when expansion is already archived."""
        existing = _make_expansion(
            expansion_id=expansion_id,
            game_id=game_id,
            archived_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        mock_expansion_repo.get_by_id.return_value = existing

        with pytest.raises(ConflictError, match="already archived"):
            await service.archive_expansion(game_id, expansion_id)

        mock_expansion_repo.update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_archive_expansion_not_found(
        self,
        service: ExpansionService,
        mock_expansion_repo: AsyncMock,
        game_id: uuid.UUID,
        expansion_id: uuid.UUID,
    ) -> None:
        """Raises NotFoundError when expansion does not exist."""
        mock_expansion_repo.get_by_id.return_value = None

        with pytest.raises(NotFoundError, match="Expansion"):
            await service.archive_expansion(game_id, expansion_id)


# ---------------------------------------------------------------------------
# restore_expansion tests
# ---------------------------------------------------------------------------


class TestRestoreExpansion:
    """Tests for ExpansionService.restore_expansion."""

    @pytest.mark.asyncio
    async def test_restore_expansion_happy_path(
        self,
        service: ExpansionService,
        mock_expansion_repo: AsyncMock,
        game_id: uuid.UUID,
        expansion_id: uuid.UUID,
    ) -> None:
        """Clears archived_at on an archived expansion."""
        existing = _make_expansion(
            expansion_id=expansion_id,
            game_id=game_id,
            archived_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        mock_expansion_repo.get_by_id.return_value = existing
        mock_expansion_repo.update.return_value = existing

        result = await service.restore_expansion(game_id, expansion_id)

        assert existing.archived_at is None
        mock_expansion_repo.update.assert_awaited_once_with(existing)
        assert result is existing

    @pytest.mark.asyncio
    async def test_restore_expansion_not_archived(
        self,
        service: ExpansionService,
        mock_expansion_repo: AsyncMock,
        game_id: uuid.UUID,
        expansion_id: uuid.UUID,
    ) -> None:
        """Raises ConflictError when expansion is not archived."""
        existing = _make_expansion(
            expansion_id=expansion_id,
            game_id=game_id,
            archived_at=None,
        )
        mock_expansion_repo.get_by_id.return_value = existing

        with pytest.raises(ConflictError, match="not archived"):
            await service.restore_expansion(game_id, expansion_id)

        mock_expansion_repo.update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_restore_expansion_not_found(
        self,
        service: ExpansionService,
        mock_expansion_repo: AsyncMock,
        game_id: uuid.UUID,
        expansion_id: uuid.UUID,
    ) -> None:
        """Raises NotFoundError when expansion does not exist."""
        mock_expansion_repo.get_by_id.return_value = None

        with pytest.raises(NotFoundError, match="Expansion"):
            await service.restore_expansion(game_id, expansion_id)


# ---------------------------------------------------------------------------
# Game silo isolation tests (cross-cutting)
# ---------------------------------------------------------------------------


class TestGameSiloIsolation:
    """Tests verifying game_id scoping across all operations."""

    @pytest.mark.asyncio
    async def test_get_passes_game_id_to_repo(
        self,
        service: ExpansionService,
        mock_expansion_repo: AsyncMock,
        expansion_id: uuid.UUID,
    ) -> None:
        """get_expansion passes game_id to repo for scoped lookup."""
        target_game_id = uuid.uuid4()
        mock_expansion_repo.get_by_id.return_value = _make_expansion()

        await service.get_expansion(target_game_id, expansion_id)

        mock_expansion_repo.get_by_id.assert_awaited_once_with(target_game_id, expansion_id)

    @pytest.mark.asyncio
    async def test_update_scopes_to_game(
        self,
        service: ExpansionService,
        mock_expansion_repo: AsyncMock,
        expansion_id: uuid.UUID,
    ) -> None:
        """update_expansion fetches expansion scoped to game_id."""
        target_game_id = uuid.uuid4()
        mock_expansion_repo.get_by_id.return_value = None

        with pytest.raises(NotFoundError):
            await service.update_expansion(target_game_id, expansion_id, ExpansionUpdate(name="X"))

        mock_expansion_repo.get_by_id.assert_awaited_once_with(target_game_id, expansion_id)

    @pytest.mark.asyncio
    async def test_archive_scopes_to_game(
        self,
        service: ExpansionService,
        mock_expansion_repo: AsyncMock,
        expansion_id: uuid.UUID,
    ) -> None:
        """archive_expansion fetches expansion scoped to game_id."""
        target_game_id = uuid.uuid4()
        mock_expansion_repo.get_by_id.return_value = None

        with pytest.raises(NotFoundError):
            await service.archive_expansion(target_game_id, expansion_id)

        mock_expansion_repo.get_by_id.assert_awaited_once_with(target_game_id, expansion_id)

    @pytest.mark.asyncio
    async def test_restore_scopes_to_game(
        self,
        service: ExpansionService,
        mock_expansion_repo: AsyncMock,
        expansion_id: uuid.UUID,
    ) -> None:
        """restore_expansion fetches expansion scoped to game_id."""
        target_game_id = uuid.uuid4()
        mock_expansion_repo.get_by_id.return_value = None

        with pytest.raises(NotFoundError):
            await service.restore_expansion(target_game_id, expansion_id)

        mock_expansion_repo.get_by_id.assert_awaited_once_with(target_game_id, expansion_id)

    @pytest.mark.asyncio
    async def test_list_passes_game_id_to_repo(
        self,
        service: ExpansionService,
        mock_expansion_repo: AsyncMock,
        mock_game_repo: AsyncMock,
    ) -> None:
        """list_expansions passes game_id to repo for scoped query."""
        from tabletop_oracle.api.pagination import PaginationParams

        target_game_id = uuid.uuid4()
        mock_game_repo.get_by_id.return_value = _make_game(game_id=target_game_id)
        mock_expansion_repo.list_by_game.return_value = ([], 0)

        filters = ExpansionFilters()
        pagination = PaginationParams(page=1, page_size=25)

        await service.list_expansions(target_game_id, filters, pagination)

        mock_expansion_repo.list_by_game.assert_awaited_once_with(
            target_game_id, filters, pagination
        )
