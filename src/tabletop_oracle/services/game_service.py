"""Business logic for Game entity management.

Encapsulates validation, tag normalisation, user attribution, archive/restore
logic, and repository delegation for the games catalogue. All domain errors
use the F001 exception hierarchy.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from tabletop_oracle.errors.exceptions import ConflictError, NotFoundError, ValidationError

if TYPE_CHECKING:
    import uuid

    from tabletop_oracle.api.pagination import PaginationParams
    from tabletop_oracle.models.game import Game
    from tabletop_oracle.repositories.game_repository import GameRepository
    from tabletop_oracle.schemas.game import GameCreate, GameFilters, GameUpdate


class GameService:
    """Business logic for Game entity management.

    Orchestrates validation, tag normalisation, user attribution, and
    repository calls for game CRUD, archive/restore, and tag discovery.

    Args:
        game_repo: Repository for game data access operations.
    """

    def __init__(self, game_repo: GameRepository) -> None:
        self._repo = game_repo

    async def create_game(self, data: GameCreate, user_id: uuid.UUID) -> Game:
        """Create a new game with validated input and normalised tags.

        Sets ``created_by`` from the authenticated user (never from client
        input). Tags are normalised (lowercased, trimmed, deduplicated,
        sorted) by the Pydantic schema before reaching the service layer.

        Args:
            data: Validated game creation schema.
            user_id: UUID of the authenticated curator.

        Returns:
            The persisted Game entity with tags set.

        Raises:
            ValidationError: If player count constraints are violated.
        """
        from tabletop_oracle.models.game import Game as GameModel

        self._validate_player_counts(data.min_players, data.max_players)
        tags = self._normalise_tags(data.tags)

        game = GameModel(
            created_by=user_id,
            name=data.name,
            publisher=data.publisher,
            year_published=data.year_published,
            edition=data.edition,
            min_players=data.min_players,
            max_players=data.max_players,
            description=data.description,
            cover_image_url=data.cover_image_url,
            complexity=self._to_orm_complexity(data.complexity),
        )

        game = await self._repo.create(game)
        if tags:
            await self._repo.set_tags(game.id, tags)

        return game

    async def get_game_detail(self, game_id: uuid.UUID) -> dict[str, Any]:
        """Fetch a game with expansions, tags, and aggregated counts.

        Returns the full detail needed for the GameDetailResponse, including
        inline expansions and document/expansion counts.

        Args:
            game_id: UUID of the game to retrieve.

        Returns:
            Dict with keys ``game``, ``expansions``, ``document_count``,
            ``expansion_count``, and ``tags``.

        Raises:
            NotFoundError: If no game exists with the given ID.
        """
        detail = await self._repo.get_detail(game_id)
        if detail is None:
            raise NotFoundError("Game", str(game_id))
        return detail

    async def get_game(self, game_id: uuid.UUID) -> Game:
        """Fetch a game by ID or raise NotFoundError.

        Args:
            game_id: UUID of the game to retrieve.

        Returns:
            The Game entity with tags eagerly loaded.

        Raises:
            NotFoundError: If no game exists with the given ID.
        """
        game = await self._repo.get_by_id(game_id)
        if game is None:
            raise NotFoundError("Game", str(game_id))
        return game

    async def list_games(
        self,
        filters: GameFilters,
        pagination: PaginationParams,
    ) -> tuple[list[dict[str, Any]], int]:
        """List games with filtering, sorting, and pagination.

        Delegates entirely to the repository, which handles dynamic query
        construction and summary aggregation.

        Args:
            filters: Parsed filter/sort parameters.
            pagination: Page number and page size.

        Returns:
            Tuple of (list of game summary dicts, total matching count).
        """
        return await self._repo.list_with_summary(filters, pagination)

    async def update_game(self, game_id: uuid.UUID, data: GameUpdate) -> Game:
        """Apply a partial update to an existing game.

        Only fields that are not UNSET are applied. Tags are normalised
        when provided. Player count consistency is validated when either
        player count field is updated.

        Args:
            game_id: UUID of the game to update.
            data: Partial update schema with UNSET-aware fields.

        Returns:
            The updated Game entity.

        Raises:
            NotFoundError: If no game exists with the given ID.
            ValidationError: If player count constraints are violated.
        """
        game = await self.get_game(game_id)
        set_fields = data.get_set_fields()

        if not set_fields:
            return game

        # Resolve effective player counts for validation
        effective_min = set_fields.get("min_players", game.min_players)
        effective_max = set_fields.get("max_players", game.max_players)

        if "min_players" in set_fields or "max_players" in set_fields:
            self._validate_player_counts(effective_min, effective_max)

        # Apply scalar fields to the game entity; extract tags for separate handling
        has_tags = "tags" in set_fields
        tags_value: list[str] | None = set_fields.pop("tags", None)

        for field_name, value in set_fields.items():
            if field_name == "complexity":
                value = self._to_orm_complexity(value)
            setattr(game, field_name, value)

        game = await self._repo.update(game)

        # Handle tags separately via set_tags
        if has_tags:
            normalised = self._normalise_tags(tags_value or [])
            await self._repo.set_tags(game_id, normalised)

        return game

    async def archive_game(self, game_id: uuid.UUID) -> Game:
        """Soft-delete a game by setting archived_at.

        Args:
            game_id: UUID of the game to archive.

        Returns:
            The updated Game entity with archived_at set.

        Raises:
            NotFoundError: If no game exists with the given ID.
            ConflictError: If the game is already archived.
        """
        game = await self.get_game(game_id)

        if game.archived_at is not None:
            raise ConflictError("Game is already archived")

        game.archived_at = datetime.now(UTC)
        return await self._repo.update(game)

    async def restore_game(self, game_id: uuid.UUID) -> Game:
        """Restore an archived game by clearing archived_at.

        Args:
            game_id: UUID of the game to restore.

        Returns:
            The updated Game entity with archived_at cleared.

        Raises:
            NotFoundError: If no game exists with the given ID.
            ConflictError: If the game is not archived.
        """
        game = await self.get_game(game_id)

        if game.archived_at is None:
            raise ConflictError("Game is not archived")

        game.archived_at = None
        return await self._repo.update(game)

    async def get_tags(self) -> list[dict[str, Any]]:
        """Get all tags with usage counts across active games.

        Returns:
            List of dicts with ``tag`` and ``count`` keys, sorted by count
            descending then tag name ascending.
        """
        return await self._repo.get_tags_with_counts()

    @staticmethod
    def _normalise_tags(tags: list[str]) -> list[str]:
        """Lowercase, strip, deduplicate, and sort tags.

        Empty and whitespace-only tags are filtered out.

        Args:
            tags: Raw tag strings.

        Returns:
            Sorted list of unique, normalised tag strings.
        """
        seen: set[str] = set()
        normalised: list[str] = []
        for tag in tags:
            clean = tag.strip().lower()
            if clean and clean not in seen:
                seen.add(clean)
                normalised.append(clean)
        return sorted(normalised)

    @staticmethod
    def _validate_player_counts(
        min_players: int | None,
        max_players: int | None,
    ) -> None:
        """Validate player count consistency.

        Both must be provided together, and min must be <= max.

        Args:
            min_players: Minimum player count (or None).
            max_players: Maximum player count (or None).

        Raises:
            ValidationError: If constraints are violated.
        """
        if (min_players is None) != (max_players is None):
            raise ValidationError("Both min_players and max_players must be provided together.")
        if min_players is not None and max_players is not None and min_players > max_players:
            raise ValidationError("min_players must be <= max_players.")

    @staticmethod
    def _to_orm_complexity(
        complexity: Any,
    ) -> Any:
        """Convert schema complexity enum to ORM enum if needed.

        The Pydantic schema uses GameComplexityFilter (StrEnum) while the
        ORM model uses GameComplexity (enum.Enum). This bridges the gap.

        Args:
            complexity: A GameComplexityFilter value, None, or UNSET.

        Returns:
            The corresponding GameComplexity ORM enum value, or None.
        """
        if complexity is None:
            return None

        from tabletop_oracle.models.enums import GameComplexity
        from tabletop_oracle.schemas.game import GameComplexityFilter

        if isinstance(complexity, GameComplexityFilter):
            return GameComplexity(complexity.value)
        return complexity
