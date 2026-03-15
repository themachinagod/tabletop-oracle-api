"""Data access for Expansion entities.

Provides CRUD operations scoped to a parent game, dynamic filtering by
archived status, pagination with total count, and document count
aggregation. All queries enforce game silo isolation — an expansion is
only accessible via its owning game_id.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import Select, func, select

from tabletop_oracle.models.document import Document
from tabletop_oracle.models.enums import DocumentStatus
from tabletop_oracle.models.expansion import Expansion
from tabletop_oracle.repositories.base import BaseRepository

if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

    from tabletop_oracle.api.pagination import PaginationParams
    from tabletop_oracle.schemas.expansion import ExpansionFilters


class ExpansionRepository(BaseRepository[Expansion]):
    """Data access for Expansion entities scoped to a parent game.

    All read queries enforce game_id scoping as a security boundary —
    an expansion belonging to game A is invisible when queried via game B.

    Args:
        session: SQLAlchemy async session for database operations.
    """

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Expansion)

    async def get_by_id(  # type: ignore[override]
        self,
        game_id: uuid.UUID,
        expansion_id: uuid.UUID,
    ) -> Expansion | None:
        """Fetch a single expansion, scoped to the given game.

        Returns None if the expansion does not exist OR belongs to a
        different game. This enforces game silo isolation.

        Args:
            game_id: UUID of the parent game (silo boundary).
            expansion_id: UUID of the expansion to retrieve.

        Returns:
            The Expansion entity, or None if not found or wrong game.
        """
        stmt = select(Expansion).where(
            Expansion.id == expansion_id,
            Expansion.game_id == game_id,
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_game(
        self,
        game_id: uuid.UUID,
        filters: ExpansionFilters,
        pagination: PaginationParams,
    ) -> tuple[list[dict[str, Any]], int]:
        """List expansions for a game with filtering, pagination, and doc counts.

        Returns expansions belonging to the specified game, optionally
        filtered by archived status. Each result includes a document_count
        of processed documents. Results are sorted by name ascending.

        Args:
            game_id: UUID of the parent game (silo boundary).
            filters: Parsed filter parameters (archived flag).
            pagination: Page number and page size.

        Returns:
            Tuple of (list of expansion dicts with document_count, total count).
        """
        # Base filter query for counting
        base_query = select(Expansion).where(Expansion.game_id == game_id)
        base_query = self._apply_filters(base_query, filters)

        total = await self._count(base_query)

        # Document count subquery — only processed docs scoped to expansion
        doc_subquery = (
            select(
                Document.expansion_id,
                func.count().label("doc_count"),
            )
            .where(Document.status == DocumentStatus.PROCESSED)
            .group_by(Document.expansion_id)
            .subquery()
        )

        # Summary query with LEFT JOIN for document counts
        summary_query = (
            select(
                Expansion,
                func.coalesce(doc_subquery.c.doc_count, 0).label("document_count"),
            )
            .outerjoin(doc_subquery, doc_subquery.c.expansion_id == Expansion.id)
            .where(Expansion.game_id == game_id)
        )

        summary_query = self._apply_filters(summary_query, filters)

        # Group by expansion + aggregation column
        summary_query = summary_query.group_by(
            Expansion.id,
            doc_subquery.c.doc_count,
        )

        # Sort by name ascending (default per design doc)
        summary_query = summary_query.order_by(Expansion.name.asc())

        # Pagination
        summary_query = summary_query.offset(pagination.offset).limit(
            pagination.page_size,
        )

        result = await self._session.execute(summary_query)
        rows = result.all()

        summaries: list[dict[str, Any]] = []
        for row in rows:
            expansion: Expansion = row[0]
            doc_count: int = row[1]
            summaries.append(
                {
                    "expansion": expansion,
                    "document_count": doc_count,
                }
            )

        return summaries, total

    async def update(self, expansion: Expansion) -> Expansion:
        """Persist changes to an existing expansion entity.

        Merges the expansion into the current session and flushes to
        generate any server-side defaults (e.g. updated_at trigger).

        Args:
            expansion: The Expansion entity with updated attributes.

        Returns:
            The merged Expansion entity.
        """
        merged = await self._session.merge(expansion)
        await self._session.flush()
        return merged

    async def _count(self, query: Select[Any]) -> int:
        """Execute a COUNT query derived from the given SELECT.

        Args:
            query: Base SELECT statement to count results for.

        Returns:
            Total number of matching rows.
        """
        count_query = select(func.count()).select_from(query.subquery())
        result = await self._session.execute(count_query)
        total: int = result.scalar_one()
        return total

    def _apply_filters(
        self,
        query: Select[Any],
        filters: ExpansionFilters,
    ) -> Select[Any]:
        """Apply dynamic WHERE clauses based on filter parameters.

        Currently supports archived status filtering. The game_id scope
        is applied by the caller, not here.

        Args:
            query: The base SELECT statement to augment.
            filters: Parsed filter parameters.

        Returns:
            Query with applicable WHERE clauses added.
        """
        if not filters.archived:
            query = query.where(Expansion.archived_at.is_(None))

        return query
