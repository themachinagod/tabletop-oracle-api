"""Data access for Knowledge Graph entities.

Provides repository classes for KGConcept, KGConceptSource, KGAssociation,
KGAssociationSource, and KGAuditLog. All queries enforce game_id scoping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from tabletop_oracle.models.knowledge_graph import (
    KGAssociation,
    KGAssociationSource,
    KGAuditLog,
    KGConcept,
    KGConceptSource,
)

if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession


class KGConceptRepository:
    """Data access for KGConcept entities scoped to a game.

    Provides concept lookup by name (including alias resolution),
    creation, and bulk queries needed by the integration service.

    Args:
        session: SQLAlchemy async session for database operations.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, concept_id: uuid.UUID) -> KGConcept | None:
        """Fetch a concept by primary key.

        Args:
            concept_id: UUID of the concept.

        Returns:
            The KGConcept entity, or None if not found.
        """
        return await self._session.get(KGConcept, concept_id)

    async def find_by_name(
        self,
        game_id: uuid.UUID,
        name: str,
    ) -> KGConcept | None:
        """Find a concept by canonical name within a game.

        Case-insensitive match on the concept name. Returns the concept
        if it exists, regardless of whether it is canonical or an alias.

        Args:
            game_id: UUID of the owning game.
            name: Concept name to search for.

        Returns:
            The matching KGConcept, or None.
        """
        stmt = select(KGConcept).where(
            KGConcept.game_id == game_id,
            KGConcept.name.ilike(name),
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_game(
        self,
        game_id: uuid.UUID,
    ) -> list[KGConcept]:
        """List all canonical concepts for a game.

        Excludes alias concepts (those with a non-null canonical_id).

        Args:
            game_id: UUID of the owning game.

        Returns:
            List of canonical KGConcept entities.
        """
        stmt = select(KGConcept).where(
            KGConcept.game_id == game_id,
            KGConcept.canonical_id.is_(None),
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def create(self, concept: KGConcept) -> KGConcept:
        """Persist a new concept.

        Args:
            concept: The KGConcept entity to create.

        Returns:
            The persisted entity.
        """
        self._session.add(concept)
        await self._session.flush()
        return concept


class KGConceptSourceRepository:
    """Data access for KGConceptSource entities.

    Args:
        session: SQLAlchemy async session for database operations.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def find_by_concept_and_chunk(
        self,
        concept_id: uuid.UUID,
        chunk_id: uuid.UUID,
    ) -> KGConceptSource | None:
        """Check if a source already exists for a concept-chunk pair.

        Args:
            concept_id: UUID of the concept.
            chunk_id: UUID of the source chunk.

        Returns:
            The existing source, or None.
        """
        stmt = select(KGConceptSource).where(
            KGConceptSource.concept_id == concept_id,
            KGConceptSource.chunk_id == chunk_id,
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_concept(
        self,
        concept_id: uuid.UUID,
    ) -> list[KGConceptSource]:
        """List all sources for a concept.

        Args:
            concept_id: UUID of the concept.

        Returns:
            List of KGConceptSource entities.
        """
        stmt = select(KGConceptSource).where(
            KGConceptSource.concept_id == concept_id,
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def create(self, source: KGConceptSource) -> KGConceptSource:
        """Persist a new concept source.

        Args:
            source: The KGConceptSource entity to create.

        Returns:
            The persisted entity.
        """
        self._session.add(source)
        await self._session.flush()
        return source

    async def update_authoritative(
        self,
        concept_id: uuid.UUID,
        authoritative_source_id: uuid.UUID,
    ) -> None:
        """Set one source as authoritative, demote all others.

        Args:
            concept_id: UUID of the concept.
            authoritative_source_id: UUID of the source to promote.
        """
        sources = await self.list_by_concept(concept_id)
        for source in sources:
            source.is_authoritative = source.id == authoritative_source_id
        await self._session.flush()


class KGAssociationRepository:
    """Data access for KGAssociation and KGAssociationSource entities.

    Args:
        session: SQLAlchemy async session for database operations.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def find_existing(
        self,
        game_id: uuid.UUID,
        source_concept_id: uuid.UUID,
        target_concept_id: uuid.UUID,
        relationship_label: str,
    ) -> KGAssociation | None:
        """Find an existing association by its unique key.

        Args:
            game_id: UUID of the owning game.
            source_concept_id: UUID of the source concept.
            target_concept_id: UUID of the target concept.
            relationship_label: Semantic label of the relationship.

        Returns:
            The existing association, or None.
        """
        stmt = select(KGAssociation).where(
            KGAssociation.game_id == game_id,
            KGAssociation.source_concept_id == source_concept_id,
            KGAssociation.target_concept_id == target_concept_id,
            KGAssociation.relationship_label == relationship_label,
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def create(self, association: KGAssociation) -> KGAssociation:
        """Persist a new association.

        Args:
            association: The KGAssociation entity to create.

        Returns:
            The persisted entity.
        """
        self._session.add(association)
        await self._session.flush()
        return association

    async def add_source(self, source: KGAssociationSource) -> KGAssociationSource:
        """Persist a new association source.

        Args:
            source: The KGAssociationSource entity to create.

        Returns:
            The persisted entity.
        """
        self._session.add(source)
        await self._session.flush()
        return source

    async def find_source_by_chunk(
        self,
        association_id: uuid.UUID,
        chunk_id: uuid.UUID,
    ) -> KGAssociationSource | None:
        """Check if an association source already exists for a chunk.

        Args:
            association_id: UUID of the association.
            chunk_id: UUID of the source chunk.

        Returns:
            The existing source, or None.
        """
        stmt = select(KGAssociationSource).where(
            KGAssociationSource.association_id == association_id,
            KGAssociationSource.chunk_id == chunk_id,
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()


class KGAuditLogRepository:
    """Data access for KGAuditLog entries.

    Append-only log of KG operations.

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
