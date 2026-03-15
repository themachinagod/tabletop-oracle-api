"""Data access for KG audit log entries.

Provides append-only write access and game-scoped query access to the
``kg_audit_log`` table. All queries enforce ``game_id`` scoping.

Design reference: docs/architecture/epic-003-knowledge-graph-engine/design.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import select
from sqlalchemy.sql import func

from tabletop_oracle.models.knowledge_graph import KGAuditLog

if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession


class KGAuditRepository:
    """Data access for KGAuditLog entries.

    Append-only log of KG operations with game-scoped queries.

    Args:
        session: SQLAlchemy async session for database operations.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        game_id: uuid.UUID,
        event_type: str,
        details: dict[str, Any],
        *,
        document_id: uuid.UUID | None = None,
        concept_id: uuid.UUID | None = None,
    ) -> KGAuditLog:
        """Write an audit log entry.

        Args:
            game_id: UUID of the related game.
            event_type: Event classification string.
            details: Structured event data.
            document_id: Optional related document UUID.
            concept_id: Optional related concept UUID.

        Returns:
            The persisted audit log entry.
        """
        entry = KGAuditLog(
            game_id=game_id,
            event_type=event_type,
            document_id=document_id,
            concept_id=concept_id,
            details=details,
        )
        self._session.add(entry)
        await self._session.flush()
        return entry

    async def list_by_game(
        self,
        game_id: uuid.UUID,
        *,
        event_type: str | None = None,
        document_id: uuid.UUID | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[KGAuditLog]:
        """List audit log entries for a game, newest first.

        Args:
            game_id: UUID of the game to query.
            event_type: Optional filter by event type.
            document_id: Optional filter by related document.
            limit: Maximum number of entries to return.
            offset: Number of entries to skip.

        Returns:
            List of KGAuditLog entries ordered by created_at descending.
        """
        stmt = (
            select(KGAuditLog)
            .where(KGAuditLog.game_id == game_id)
            .order_by(KGAuditLog.created_at.desc())
            .limit(limit)
            .offset(offset)
        )

        if event_type is not None:
            stmt = stmt.where(KGAuditLog.event_type == event_type)

        if document_id is not None:
            stmt = stmt.where(KGAuditLog.document_id == document_id)

        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def count_by_game(
        self,
        game_id: uuid.UUID,
        *,
        event_type: str | None = None,
    ) -> int:
        """Count audit log entries for a game.

        Args:
            game_id: UUID of the game to query.
            event_type: Optional filter by event type.

        Returns:
            Total number of matching entries.
        """
        stmt = select(func.count(KGAuditLog.id)).where(
            KGAuditLog.game_id == game_id,
        )

        if event_type is not None:
            stmt = stmt.where(KGAuditLog.event_type == event_type)

        result = await self._session.execute(stmt)
        return result.scalar_one()
