"""Data access for Game and GameTag entities.

Provides CRUD operations, dynamic filtering with full-text search,
pagination, tag management, and aggregation queries for the games
catalogue. Built on top of BaseRepository for standard operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import Select, case, func, literal_column, select, text

from tabletop_oracle.models.document import Document
from tabletop_oracle.models.enums import DocumentStatus
from tabletop_oracle.models.expansion import Expansion
from tabletop_oracle.models.game import Game, GameTag
from tabletop_oracle.repositories.base import BaseRepository
from tabletop_oracle.repositories.query_utils import escape_like

if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

    from tabletop_oracle.api.pagination import PaginationParams
    from tabletop_oracle.schemas.game import GameFilters

# Mapping from sort parameter strings to SQLAlchemy order-by clauses.
# Each key supports an optional ``-`` prefix for descending order in the
# GameFilters schema; that prefix is already embedded in the map keys.
_SORT_MAP: dict[str, Any] = {
    "name": Game.name.asc(),
    "-name": Game.name.desc(),
    "updated_at": Game.updated_at.asc(),
    "-updated_at": Game.updated_at.desc(),
    "created_at": Game.created_at.asc(),
    "-created_at": Game.created_at.desc(),
}

_DEFAULT_SORT = Game.name.asc()


class GameRepository(BaseRepository[Game]):
    """Data access for Game and GameTag entities.

    Extends BaseRepository with game-specific query logic: dynamic
    filtering (archived, search, player_count, complexity, tags),
    sorting, pagination with total count, LEFT JOIN aggregation for
    summary counts, and atomic tag management.

    Args:
        session: SQLAlchemy async session for database operations.
    """

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Game)

    async def get_by_id(self, game_id: uuid.UUID) -> Game | None:
        """Fetch a single game with its tags eagerly loaded.

        Args:
            game_id: UUID of the game to retrieve.

        Returns:
            The Game entity with tags populated, or None if not found.
        """
        from sqlalchemy.orm import selectinload

        stmt = select(Game).options(selectinload(Game.tags)).where(Game.id == game_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_detail(self, game_id: uuid.UUID) -> dict[str, Any] | None:
        """Fetch a game with tags, expansions, and aggregated counts.

        Returns all data needed for the GameDetailResponse: the game entity,
        its expansions (all, including archived), document count (processed
        only), and expansion count (active only).

        Args:
            game_id: UUID of the game to retrieve.

        Returns:
            Dict with keys ``game``, ``expansions``, ``document_count``,
            ``expansion_count``, and ``tags``; or None if not found.
        """
        from sqlalchemy.orm import selectinload

        stmt = (
            select(Game)
            .options(selectinload(Game.tags), selectinload(Game.expansions))
            .where(Game.id == game_id)
        )
        result = await self._session.execute(stmt)
        game = result.scalar_one_or_none()
        if game is None:
            return None

        # Document count (processed only)
        doc_count_stmt = (
            select(func.count())
            .select_from(Document)
            .where(Document.game_id == game_id, Document.status == DocumentStatus.PROCESSED.value)
        )
        doc_result = await self._session.execute(doc_count_stmt)
        doc_count: int = doc_result.scalar_one()

        # Active expansion count
        active_exp_count = sum(1 for exp in game.expansions if exp.archived_at is None)

        tag_list = sorted(t.tag for t in game.tags)

        return {
            "game": game,
            "expansions": list(game.expansions),
            "document_count": doc_count,
            "expansion_count": active_exp_count,
            "tags": tag_list,
        }

    async def list_with_summary(
        self,
        filters: GameFilters,
        pagination: PaginationParams,
    ) -> tuple[list[dict[str, Any]], int]:
        """List games with dynamic filtering, sorting, pagination, and summary counts.

        Builds a query with optional filters for archived status, full-text
        search, player count, complexity, and tags. Includes LEFT JOIN
        aggregations for document_count (processed only) and expansion_count
        (active only).

        Args:
            filters: Parsed filter/sort parameters from the API layer.
            pagination: Page number and page size.

        Returns:
            Tuple of (list of game summary dicts, total matching count).
            Each dict contains all Game columns plus document_count,
            expansion_count, and tags.
        """
        # Build the base filter query for counting
        base_query = select(Game)
        base_query = self._apply_filters(base_query, filters)

        # Get total count before pagination
        total = await self._count(base_query)

        # Build the full summary query with aggregations
        doc_subquery = (
            select(
                Document.game_id,
                func.count().label("doc_count"),
            )
            .where(Document.status == DocumentStatus.PROCESSED.value)
            .group_by(Document.game_id)
            .subquery()
        )

        exp_subquery = (
            select(
                Expansion.game_id,
                func.count().label("exp_count"),
            )
            .where(Expansion.archived_at.is_(None))
            .group_by(Expansion.game_id)
            .subquery()
        )

        summary_query = (
            select(
                Game,
                func.coalesce(doc_subquery.c.doc_count, 0).label("document_count"),
                func.coalesce(exp_subquery.c.exp_count, 0).label("expansion_count"),
                func.array_agg(
                    case(
                        (GameTag.tag.is_not(None), GameTag.tag),
                        else_=literal_column("NULL"),
                    )
                ).label("tags"),
            )
            .outerjoin(doc_subquery, doc_subquery.c.game_id == Game.id)
            .outerjoin(exp_subquery, exp_subquery.c.game_id == Game.id)
            .outerjoin(GameTag, GameTag.game_id == Game.id)
        )

        # Apply the same filters to the summary query
        summary_query = self._apply_filters(summary_query, filters)

        # Group by game + aggregation columns
        summary_query = summary_query.group_by(
            Game.id,
            doc_subquery.c.doc_count,
            exp_subquery.c.exp_count,
        )

        # Sorting
        sort_clause = _SORT_MAP.get(filters.sort, _DEFAULT_SORT)
        summary_query = summary_query.order_by(sort_clause)

        # Pagination
        summary_query = summary_query.offset(pagination.offset).limit(pagination.page_size)

        result = await self._session.execute(summary_query)
        rows = result.all()

        summaries: list[dict[str, Any]] = []
        for row in rows:
            game: Game = row[0]
            doc_count: int = row[1]
            exp_count: int = row[2]
            raw_tags: list[str | None] | None = row[3]

            # Filter out NULL values from array_agg and sort
            tag_list = sorted(t for t in (raw_tags or []) if t is not None)

            summaries.append(
                {
                    "game": game,
                    "document_count": doc_count,
                    "expansion_count": exp_count,
                    "tags": tag_list,
                }
            )

        return summaries, total

    async def update(self, game: Game) -> Game:
        """Persist changes to an existing game entity.

        Merges the game into the current session and flushes to generate
        any server-side defaults (e.g. updated_at trigger).

        Args:
            game: The Game entity with updated attributes.

        Returns:
            The merged Game entity.
        """
        merged = await self._session.merge(game)
        await self._session.flush()
        return merged

    async def set_tags(self, game_id: uuid.UUID, tags: list[str]) -> None:
        """Replace all tags for a game atomically.

        Deletes existing tags and inserts the new set in a single flush.
        The caller is responsible for ensuring the game exists.

        Args:
            game_id: UUID of the game to update tags for.
            tags: New complete set of tag strings.
        """
        # Delete existing tags
        existing = await self._session.execute(select(GameTag).where(GameTag.game_id == game_id))
        for tag_entity in existing.scalars().all():
            await self._session.delete(tag_entity)

        # Insert new tags
        for tag_value in tags:
            new_tag = GameTag(game_id=game_id, tag=tag_value)
            self._session.add(new_tag)

        await self._session.flush()

    async def get_tags_with_counts(self) -> list[dict[str, Any]]:
        """Get all tags with usage counts across active (non-archived) games.

        Returns tags sorted by count descending, then tag name ascending.

        Returns:
            List of dicts with ``tag`` (str) and ``count`` (int) keys.
        """
        stmt = (
            select(
                GameTag.tag,
                func.count().label("count"),
            )
            .join(Game, Game.id == GameTag.game_id)
            .where(Game.archived_at.is_(None))
            .group_by(GameTag.tag)
            .order_by(text("count DESC"), GameTag.tag.asc())
        )

        result = await self._session.execute(stmt)
        return [{"tag": row.tag, "count": row.count} for row in result.all()]

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
        filters: GameFilters,
    ) -> Select[Any]:
        """Apply dynamic WHERE clauses based on filter parameters.

        Handles archived status, full-text search (with ILIKE fallback
        for short terms), player count range, complexity IN filter, and
        tag AND logic via correlated subqueries.

        Args:
            query: The base SELECT statement to augment.
            filters: Parsed filter parameters.

        Returns:
            Query with applicable WHERE clauses added.
        """
        # Archived filter (default: active only)
        if not filters.archived:
            query = query.where(Game.archived_at.is_(None))

        # Text search
        if filters.search:
            if len(filters.search) < 3:
                safe_term = escape_like(filters.search)
                query = query.where(Game.name.ilike(f"%{safe_term}%"))
            else:
                query = query.where(
                    func.to_tsvector("english", Game.name).bool_op("@@")(
                        func.plainto_tsquery("english", filters.search)
                    )
                )

        # Player count — game supports this count if min <= count <= max
        if filters.player_count is not None:
            query = query.where(
                Game.min_players <= filters.player_count,
                Game.max_players >= filters.player_count,
            )

        # Complexity (IN / OR logic)
        if filters.complexity:
            # Use string values directly to avoid asyncpg enum serialization issues
            complexity_values = [c.value for c in filters.complexity]
            query = query.where(Game.complexity.in_(complexity_values))

        # Tags (AND logic via correlated subqueries)
        if filters.tags:
            for tag in filters.tags:
                tag_subquery = select(GameTag.game_id).where(GameTag.tag == tag)
                query = query.where(Game.id.in_(tag_subquery))

        return query
