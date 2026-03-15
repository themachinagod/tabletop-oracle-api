"""Data access for Document entities.

Provides CRUD operations scoped to a parent game, dynamic filtering by
status/type/expansion/archived status, sorting, pagination with total
count, and aggregation queries for detail views. All queries enforce
game silo isolation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sqlalchemy import Select, func, select

from tabletop_oracle.models.document import Document, DocumentChunk, DocumentVersion
from tabletop_oracle.models.enums import DocumentStatus, DocumentType

if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

    from tabletop_oracle.api.pagination import PaginationParams
    from tabletop_oracle.schemas.document import DocumentFilters

logger = logging.getLogger(__name__)

# Mapping from sort parameter strings to SQLAlchemy order-by clauses.
_SORT_MAP: dict[str, Any] = {
    "name": Document.name.asc(),
    "-name": Document.name.desc(),
    "uploaded_at": Document.uploaded_at.asc(),
    "-uploaded_at": Document.uploaded_at.desc(),
    "status": Document.status.asc(),
    "-status": Document.status.desc(),
}

_DEFAULT_SORT = Document.uploaded_at.desc()


class DocumentRepository:
    """Data access for Document entities scoped to a parent game.

    All read queries enforce game_id scoping as a security boundary.
    Does not extend BaseRepository because Document inherits from
    MappedBase (not Base) and all queries are game-scoped.

    Args:
        session: SQLAlchemy async session for database operations.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, document: Document) -> Document:
        """Persist a new document entity.

        Args:
            document: The Document entity to create.

        Returns:
            The persisted Document entity.
        """
        self._session.add(document)
        await self._session.flush()
        return document

    async def get_by_id(
        self,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> Document | None:
        """Fetch a single document scoped to the given game.

        Returns None if the document does not exist OR belongs to a
        different game. Excludes archived documents.

        Args:
            game_id: UUID of the parent game (silo boundary).
            document_id: UUID of the document to retrieve.

        Returns:
            The Document entity, or None if not found or wrong game.
        """
        stmt = select(Document).where(
            Document.id == document_id,
            Document.game_id == game_id,
            Document.archived_at.is_(None),
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_detail(
        self,
        game_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> dict[str, Any] | None:
        """Fetch document with chunk_count and version_count.

        Args:
            game_id: UUID of the parent game (silo boundary).
            document_id: UUID of the document to retrieve.

        Returns:
            Dict with document, chunk_count, version_count, or None.
        """
        doc = await self.get_by_id(game_id, document_id)
        if doc is None:
            return None

        chunk_count = await self._count_chunks(document_id)
        version_count = await self._count_versions(document_id)

        return {
            "document": doc,
            "chunk_count": chunk_count,
            "version_count": version_count,
        }

    async def list_by_game(
        self,
        game_id: uuid.UUID,
        filters: DocumentFilters,
        pagination: PaginationParams,
    ) -> tuple[list[Document], int]:
        """List documents for a game with filtering, sorting, and pagination.

        Args:
            game_id: UUID of the parent game (silo boundary).
            filters: Parsed filter parameters.
            pagination: Page number and page size.

        Returns:
            Tuple of (list of Document entities, total count).
        """
        base_query = select(Document).where(
            Document.game_id == game_id,
            Document.archived_at.is_(None),
        )
        base_query = self._apply_filters(base_query, filters)

        total = await self._count(base_query)

        sort_clause = _SORT_MAP.get(filters.sort, _DEFAULT_SORT)
        query = base_query.order_by(sort_clause)
        query = query.offset(pagination.offset).limit(pagination.page_size)

        result = await self._session.execute(query)
        documents = list(result.scalars().all())

        return documents, total

    async def update(self, document: Document) -> Document:
        """Persist changes to an existing document entity.

        Args:
            document: The Document entity with updated attributes.

        Returns:
            The merged Document entity.
        """
        merged = await self._session.merge(document)
        await self._session.flush()
        return merged

    async def create_version(self, version: DocumentVersion) -> DocumentVersion:
        """Persist a new document version.

        Args:
            version: The DocumentVersion entity to create.

        Returns:
            The persisted DocumentVersion entity.
        """
        self._session.add(version)
        await self._session.flush()
        return version

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

    async def _count_chunks(self, document_id: uuid.UUID) -> int:
        """Count chunks for a document.

        Args:
            document_id: UUID of the document.

        Returns:
            Number of chunks.
        """
        stmt = select(func.count()).where(DocumentChunk.document_id == document_id)
        result = await self._session.execute(stmt)
        return result.scalar_one()

    async def _count_versions(self, document_id: uuid.UUID) -> int:
        """Count versions for a document.

        Args:
            document_id: UUID of the document.

        Returns:
            Number of versions.
        """
        stmt = select(func.count()).where(DocumentVersion.document_id == document_id)
        result = await self._session.execute(stmt)
        return result.scalar_one()

    def _apply_filters(
        self,
        query: Select[Any],
        filters: DocumentFilters,
    ) -> Select[Any]:
        """Apply dynamic WHERE clauses based on filter parameters.

        Args:
            query: The base SELECT statement to augment.
            filters: Parsed filter parameters.

        Returns:
            Query with applicable WHERE clauses added.
        """
        if filters.status:
            orm_values = [DocumentStatus(s.value) for s in filters.status]
            query = query.where(Document.status.in_(orm_values))

        if filters.type:
            orm_values_type = [DocumentType(t.value) for t in filters.type]
            query = query.where(Document.type.in_(orm_values_type))

        if filters.expansion_id is not None:
            if filters.expansion_id == "null":
                query = query.where(Document.expansion_id.is_(None))
            else:
                query = query.where(Document.expansion_id == filters.expansion_id)

        return query
