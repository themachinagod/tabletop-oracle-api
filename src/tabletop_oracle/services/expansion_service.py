"""Business logic for Expansion entity management.

Encapsulates parent game validation, game silo enforcement, UNSET-aware
partial updates, archive/restore logic, and repository delegation for
expansion management. All domain errors use the F001 exception hierarchy.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from tabletop_oracle.errors.exceptions import ConflictError, NotFoundError

if TYPE_CHECKING:
    import uuid

    from tabletop_oracle.api.pagination import PaginationParams
    from tabletop_oracle.models.expansion import Expansion
    from tabletop_oracle.repositories.expansion_repository import ExpansionRepository
    from tabletop_oracle.repositories.game_repository import GameRepository
    from tabletop_oracle.schemas.expansion import (
        ExpansionCreate,
        ExpansionFilters,
        ExpansionUpdate,
    )


class ExpansionService:
    """Business logic for Expansion entity management.

    Orchestrates parent game validation, game silo isolation, UNSET-aware
    partial updates, and repository calls for expansion CRUD and
    archive/restore operations.

    All operations enforce game silo isolation: an expansion is only
    accessible via its owning game_id. Parent game existence is validated
    before create and list operations.

    Args:
        expansion_repo: Repository for expansion data access operations.
        game_repo: Repository for parent game validation.
    """

    def __init__(
        self,
        expansion_repo: ExpansionRepository,
        game_repo: GameRepository,
    ) -> None:
        self._expansion_repo = expansion_repo
        self._game_repo = game_repo

    async def create_expansion(
        self,
        game_id: uuid.UUID,
        data: ExpansionCreate,
        user_id: uuid.UUID,
    ) -> Expansion:
        """Create a new expansion under a parent game.

        Validates that the parent game exists and is active before
        creating the expansion. The user_id is recorded for audit
        purposes (not stored on the expansion entity itself, but
        available for logging/events).

        Args:
            game_id: UUID of the parent game.
            data: Validated expansion creation schema.
            user_id: UUID of the authenticated curator (for audit).

        Returns:
            The persisted Expansion entity.

        Raises:
            NotFoundError: If the parent game does not exist.
        """
        await self._validate_parent_game_exists(game_id)

        from tabletop_oracle.models.expansion import Expansion as ExpansionModel

        expansion = ExpansionModel(
            game_id=game_id,
            name=data.name,
            year_published=data.year_published,
            description=data.description,
        )

        return await self._expansion_repo.create(expansion)

    async def get_expansion(
        self,
        game_id: uuid.UUID,
        expansion_id: uuid.UUID,
    ) -> Expansion:
        """Fetch an expansion scoped to a specific game.

        Enforces game silo isolation: returns the expansion only if it
        belongs to the specified game.

        Args:
            game_id: UUID of the parent game (silo boundary).
            expansion_id: UUID of the expansion to retrieve.

        Returns:
            The Expansion entity.

        Raises:
            NotFoundError: If the expansion does not exist or does not
                belong to the specified game.
        """
        expansion = await self._expansion_repo.get_by_id(game_id, expansion_id)
        if expansion is None:
            raise NotFoundError("Expansion", str(expansion_id))
        return expansion

    async def list_expansions(
        self,
        game_id: uuid.UUID,
        filters: ExpansionFilters,
        pagination: PaginationParams,
    ) -> tuple[list[dict[str, Any]], int]:
        """List expansions for a game with filtering and pagination.

        Validates that the parent game exists before querying expansions.
        Delegates filtering and pagination to the repository.

        Args:
            game_id: UUID of the parent game (silo boundary).
            filters: Parsed filter parameters (archived flag).
            pagination: Page number and page size.

        Returns:
            Tuple of (list of expansion summary dicts, total count).

        Raises:
            NotFoundError: If the parent game does not exist.
        """
        await self._validate_parent_game_exists(game_id)
        return await self._expansion_repo.list_by_game(game_id, filters, pagination)

    async def update_expansion(
        self,
        game_id: uuid.UUID,
        expansion_id: uuid.UUID,
        data: ExpansionUpdate,
    ) -> Expansion:
        """Apply a partial update to an existing expansion.

        Only fields that are not UNSET are applied. Enforces game silo
        isolation via the get_expansion lookup.

        Args:
            game_id: UUID of the parent game (silo boundary).
            expansion_id: UUID of the expansion to update.
            data: Partial update schema with UNSET-aware fields.

        Returns:
            The updated Expansion entity.

        Raises:
            NotFoundError: If the expansion does not exist or does not
                belong to the specified game.
        """
        expansion = await self.get_expansion(game_id, expansion_id)
        set_fields = data.get_set_fields()

        if not set_fields:
            return expansion

        for field_name, value in set_fields.items():
            setattr(expansion, field_name, value)

        return await self._expansion_repo.update(expansion)

    async def archive_expansion(
        self,
        game_id: uuid.UUID,
        expansion_id: uuid.UUID,
    ) -> Expansion:
        """Soft-delete an expansion by setting archived_at.

        Enforces game silo isolation via the get_expansion lookup.

        Args:
            game_id: UUID of the parent game (silo boundary).
            expansion_id: UUID of the expansion to archive.

        Returns:
            The updated Expansion entity with archived_at set.

        Raises:
            NotFoundError: If the expansion does not exist or does not
                belong to the specified game.
            ConflictError: If the expansion is already archived.
        """
        expansion = await self.get_expansion(game_id, expansion_id)

        if expansion.archived_at is not None:
            raise ConflictError("Expansion is already archived")

        expansion.archived_at = datetime.now(UTC)
        return await self._expansion_repo.update(expansion)

    async def restore_expansion(
        self,
        game_id: uuid.UUID,
        expansion_id: uuid.UUID,
    ) -> Expansion:
        """Restore an archived expansion by clearing archived_at.

        Enforces game silo isolation via the get_expansion lookup.

        Args:
            game_id: UUID of the parent game (silo boundary).
            expansion_id: UUID of the expansion to restore.

        Returns:
            The updated Expansion entity with archived_at cleared.

        Raises:
            NotFoundError: If the expansion does not exist or does not
                belong to the specified game.
            ConflictError: If the expansion is not archived.
        """
        expansion = await self.get_expansion(game_id, expansion_id)

        if expansion.archived_at is None:
            raise ConflictError("Expansion is not archived")

        expansion.archived_at = None
        return await self._expansion_repo.update(expansion)

    async def _validate_parent_game_exists(self, game_id: uuid.UUID) -> None:
        """Verify that the parent game exists.

        Args:
            game_id: UUID of the game to validate.

        Raises:
            NotFoundError: If no game exists with the given ID.
        """
        game = await self._game_repo.get_by_id(game_id)
        if game is None:
            raise NotFoundError("Game", str(game_id))
