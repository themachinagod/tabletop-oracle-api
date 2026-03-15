"""KG audit service for logging all knowledge graph operations.

Provides typed methods for logging each KG event type to the
``kg_audit_log`` table with structured metadata (AC-509). All events
include timestamps (via database ``server_default``), game references,
and event-specific detail payloads.

Design reference: docs/architecture/epic-003-knowledge-graph-engine/design.md
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID

    from tabletop_oracle.models.knowledge_graph import KGAuditLog
    from tabletop_oracle.repositories.kg_audit_repository import KGAuditRepository

logger = logging.getLogger(__name__)


# Event type constants used throughout the KG pipeline.
EVENT_EXTRACTION_COMPLETE = "extraction_complete"
EVENT_ASSOCIATIONS_DISCOVERED = "associations_discovered"
EVENT_CONFLICT_DETECTED = "conflict_detected"
EVENT_CONFLICT_RESOLVED = "conflict_resolved"
EVENT_DOCUMENT_REMOVED = "document_removed"
EVENT_GRAPH_REBUILT = "graph_rebuilt"
EVENT_RETRIEVAL_EXECUTED = "retrieval_executed"

ALL_EVENT_TYPES = frozenset(
    {
        EVENT_EXTRACTION_COMPLETE,
        EVENT_ASSOCIATIONS_DISCOVERED,
        EVENT_CONFLICT_DETECTED,
        EVENT_CONFLICT_RESOLVED,
        EVENT_DOCUMENT_REMOVED,
        EVENT_GRAPH_REBUILT,
        EVENT_RETRIEVAL_EXECUTED,
    }
)


class KGAuditService:
    """Logs KG construction and retrieval events.

    All events are persisted to the ``kg_audit_log`` table with timestamps,
    document references, and event metadata. Provides typed logging methods
    for each event type and a query interface for audit history.

    Args:
        audit_repo: Repository for KGAuditLog persistence and queries.
    """

    def __init__(self, audit_repo: KGAuditRepository) -> None:
        self._repo = audit_repo

    async def log_extraction(
        self,
        game_id: UUID,
        document_id: UUID,
        *,
        concept_count: int,
        chunk_count: int,
    ) -> KGAuditLog:
        """Log a concept extraction completion event.

        Recorded after all chunks from a document have been processed
        and concepts extracted via the LLM.

        Args:
            game_id: UUID of the owning game.
            document_id: UUID of the processed document.
            concept_count: Number of concepts extracted.
            chunk_count: Number of chunks processed.

        Returns:
            The persisted audit log entry.
        """
        entry = await self._repo.create(
            game_id=game_id,
            event_type=EVENT_EXTRACTION_COMPLETE,
            document_id=document_id,
            details={
                "concept_count": concept_count,
                "chunk_count": chunk_count,
            },
        )
        logger.info(
            "KG extraction complete: game=%s doc=%s concepts=%d chunks=%d",
            game_id,
            document_id,
            concept_count,
            chunk_count,
        )
        return entry

    async def log_association_discovery(
        self,
        game_id: UUID,
        document_id: UUID,
        *,
        association_count: int,
        cross_document_count: int,
    ) -> KGAuditLog:
        """Log an association discovery event.

        Recorded after associations are discovered between concepts
        from a document, including cross-document associations.

        Args:
            game_id: UUID of the owning game.
            document_id: UUID of the processed document.
            association_count: Total associations discovered.
            cross_document_count: Associations linking to concepts from
                other documents.

        Returns:
            The persisted audit log entry.
        """
        entry = await self._repo.create(
            game_id=game_id,
            event_type=EVENT_ASSOCIATIONS_DISCOVERED,
            document_id=document_id,
            details={
                "association_count": association_count,
                "cross_document_count": cross_document_count,
            },
        )
        logger.info(
            "KG associations discovered: game=%s doc=%s total=%d cross_doc=%d",
            game_id,
            document_id,
            association_count,
            cross_document_count,
        )
        return entry

    async def log_conflict_detected(
        self,
        game_id: UUID,
        concept_id: UUID,
        *,
        existing_source_doc: UUID,
        new_source_doc: UUID,
        resolution: str,
    ) -> KGAuditLog:
        """Log a conflict detection event.

        Recorded when a concept receives a source from a different
        document type and a priority-based resolution is determined.

        Args:
            game_id: UUID of the owning game.
            concept_id: UUID of the concept with conflicting sources.
            existing_source_doc: UUID of the existing source document.
            new_source_doc: UUID of the new conflicting document.
            resolution: Description of how the conflict was resolved.

        Returns:
            The persisted audit log entry.
        """
        entry = await self._repo.create(
            game_id=game_id,
            event_type=EVENT_CONFLICT_DETECTED,
            concept_id=concept_id,
            details={
                "existing_source_doc": str(existing_source_doc),
                "new_source_doc": str(new_source_doc),
                "resolution": resolution,
            },
        )
        logger.info(
            "KG conflict detected: game=%s concept=%s resolution=%s",
            game_id,
            concept_id,
            resolution,
        )
        return entry

    async def log_conflict_resolved(
        self,
        game_id: UUID,
        concept_id: UUID,
        *,
        new_authoritative_doc: UUID,
        previous_authoritative_doc: UUID,
        new_document_type: str,
        previous_document_type: str,
    ) -> KGAuditLog:
        """Log a conflict resolution event.

        Recorded when the authoritative source for a concept changes
        due to a higher-priority document type.

        Args:
            game_id: UUID of the owning game.
            concept_id: UUID of the concept whose authority changed.
            new_authoritative_doc: UUID of the new authoritative document.
            previous_authoritative_doc: UUID of the previous authoritative
                document.
            new_document_type: Document type of the new authoritative source.
            previous_document_type: Document type of the previous source.

        Returns:
            The persisted audit log entry.
        """
        entry = await self._repo.create(
            game_id=game_id,
            event_type=EVENT_CONFLICT_RESOLVED,
            concept_id=concept_id,
            document_id=new_authoritative_doc,
            details={
                "new_authoritative_doc": str(new_authoritative_doc),
                "previous_authoritative_doc": str(previous_authoritative_doc),
                "new_document_type": new_document_type,
                "previous_document_type": previous_document_type,
            },
        )
        logger.info(
            "KG conflict resolved: game=%s concept=%s %s(%s) supersedes %s(%s)",
            game_id,
            concept_id,
            new_document_type,
            new_authoritative_doc,
            previous_document_type,
            previous_authoritative_doc,
        )
        return entry

    async def log_document_removal(
        self,
        game_id: UUID,
        document_id: UUID,
        *,
        concepts_removed: int,
        concepts_updated: int,
    ) -> KGAuditLog:
        """Log a document removal event.

        Recorded when a document's contribution is removed from the KG.
        Concepts solely sourced from this document are removed; concepts
        with multiple sources have this document's reference removed.

        Args:
            game_id: UUID of the owning game.
            document_id: UUID of the removed document.
            concepts_removed: Number of concepts entirely removed.
            concepts_updated: Number of concepts that lost a source but
                were retained (have other sources).

        Returns:
            The persisted audit log entry.
        """
        entry = await self._repo.create(
            game_id=game_id,
            event_type=EVENT_DOCUMENT_REMOVED,
            document_id=document_id,
            details={
                "concepts_removed": concepts_removed,
                "concepts_updated": concepts_updated,
            },
        )
        logger.info(
            "KG document removed: game=%s doc=%s removed=%d updated=%d",
            game_id,
            document_id,
            concepts_removed,
            concepts_updated,
        )
        return entry

    async def log_graph_rebuilt(
        self,
        game_id: UUID,
        *,
        document_count: int,
        concept_count: int,
        association_count: int,
    ) -> KGAuditLog:
        """Log a full graph rebuild event.

        Recorded when the entire KG for a game is rebuilt from scratch
        (e.g., via the reprocess endpoint).

        Args:
            game_id: UUID of the owning game.
            document_count: Number of documents reprocessed.
            concept_count: Total concepts in the rebuilt graph.
            association_count: Total associations in the rebuilt graph.

        Returns:
            The persisted audit log entry.
        """
        entry = await self._repo.create(
            game_id=game_id,
            event_type=EVENT_GRAPH_REBUILT,
            details={
                "document_count": document_count,
                "concept_count": concept_count,
                "association_count": association_count,
            },
        )
        logger.info(
            "KG graph rebuilt: game=%s docs=%d concepts=%d associations=%d",
            game_id,
            document_count,
            concept_count,
            association_count,
        )
        return entry

    async def log_retrieval(
        self,
        game_id: UUID,
        *,
        query_text: str,
        results_count: int,
        expansion_filter_active: bool,
    ) -> KGAuditLog:
        """Log a retrieval execution event.

        Recorded when the KG retrieval service is invoked by the AI
        workflow to answer a player question.

        Args:
            game_id: UUID of the owning game.
            query_text: The search query text.
            results_count: Number of concepts returned.
            expansion_filter_active: Whether expansion filtering was applied.

        Returns:
            The persisted audit log entry.
        """
        entry = await self._repo.create(
            game_id=game_id,
            event_type=EVENT_RETRIEVAL_EXECUTED,
            details={
                "query_text": query_text,
                "results_count": results_count,
                "expansion_filter_active": expansion_filter_active,
            },
        )
        logger.debug(
            "KG retrieval executed: game=%s results=%d expansion_filter=%s",
            game_id,
            results_count,
            expansion_filter_active,
        )
        return entry

    async def get_audit_history(
        self,
        game_id: UUID,
        *,
        event_type: str | None = None,
        document_id: UUID | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[KGAuditLog]:
        """Query audit history for a game.

        Returns entries ordered by creation time descending (newest first).

        Args:
            game_id: UUID of the game to query.
            event_type: Optional filter by event type.
            document_id: Optional filter by related document.
            limit: Maximum number of entries to return.
            offset: Number of entries to skip.

        Returns:
            List of KGAuditLog entries matching the criteria.
        """
        return await self._repo.list_by_game(
            game_id,
            event_type=event_type,
            document_id=document_id,
            limit=limit,
            offset=offset,
        )

    async def get_event_count(
        self,
        game_id: UUID,
        *,
        event_type: str | None = None,
    ) -> int:
        """Count audit events for a game.

        Args:
            game_id: UUID of the game to query.
            event_type: Optional filter by event type.

        Returns:
            Total number of matching audit entries.
        """
        return await self._repo.count_by_game(
            game_id,
            event_type=event_type,
        )
