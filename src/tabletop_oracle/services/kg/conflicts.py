"""Conflict resolution for knowledge graph concept sources.

Resolves conflicts when a concept receives a new source from a
higher-priority document type. Both sources are retained, but the
higher-priority source becomes authoritative. Conflicts are logged
to the kg_audit_log table.

Design reference: docs/architecture/epic-003-knowledge-graph-engine/design.md
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID

    from tabletop_oracle.models.knowledge_graph import KGConceptSource
    from tabletop_oracle.repositories.kg_repository import (
        KGAuditLogRepository,
        KGConceptSourceRepository,
    )

logger = logging.getLogger(__name__)


# Priority order for document types. Higher value = higher authority.
# When two sources conflict, the one from a higher-priority document
# type becomes the authoritative source.
PRIORITY_ORDER: dict[str, int] = {
    "errata": 60,
    "faq": 50,
    "core_rules": 40,
    "expansion_rules": 40,
    "strategy": 20,
    "other": 10,
}

_DEFAULT_PRIORITY = 10


class ConflictResolutionService:
    """Resolves conflicts between concept sources using document type priority.

    When a new source is added to a concept and its document type has
    higher priority than the current authoritative source, the new
    source becomes authoritative. Both sources are always retained.
    Every conflict is logged to the audit trail.

    Args:
        source_repo: Repository for KGConceptSource operations.
        audit_repo: Repository for KGAuditLog operations.
    """

    def __init__(
        self,
        source_repo: KGConceptSourceRepository,
        audit_repo: KGAuditLogRepository,
    ) -> None:
        self._source_repo = source_repo
        self._audit_repo = audit_repo

    async def resolve_conflicts(
        self,
        game_id: UUID,
        concept_id: UUID,
        new_source: KGConceptSource,
    ) -> None:
        """Resolve conflicts for a concept after a new source is added.

        Compares the new source's document type priority against the
        current authoritative source. If the new source has higher
        priority, it becomes authoritative and the previous one is
        demoted. Both are always retained.

        A conflict is logged to kg_audit_log whenever the authoritative
        source changes.

        Args:
            game_id: UUID of the owning game (for audit logging).
            concept_id: UUID of the concept with the new source.
            new_source: The newly added KGConceptSource.
        """
        sources = await self._source_repo.list_by_concept(concept_id)

        if len(sources) <= 1:
            # Single source — no conflict possible
            return

        current_authoritative = self._find_authoritative(sources)
        if current_authoritative is None:
            # No current authoritative — promote the new source
            await self._source_repo.update_authoritative(concept_id, new_source.id)
            return

        if current_authoritative.id == new_source.id:
            # New source is already the authoritative one
            return

        new_priority = self._get_priority(new_source.document_type)
        current_priority = self._get_priority(current_authoritative.document_type)

        if new_priority > current_priority:
            await self._source_repo.update_authoritative(concept_id, new_source.id)
            await self._log_conflict(
                game_id=game_id,
                concept_id=concept_id,
                new_source=new_source,
                previous_source=current_authoritative,
            )
            logger.info(
                "Conflict resolved for concept %s: '%s' (%s, priority=%d) "
                "supersedes '%s' (%s, priority=%d)",
                concept_id,
                new_source.document_name,
                new_source.document_type,
                new_priority,
                current_authoritative.document_name,
                current_authoritative.document_type,
                current_priority,
            )

    @staticmethod
    def _find_authoritative(
        sources: list[KGConceptSource],
    ) -> KGConceptSource | None:
        """Find the current authoritative source.

        Args:
            sources: All sources for a concept.

        Returns:
            The authoritative source, or None if none is marked.
        """
        for source in sources:
            if source.is_authoritative:
                return source
        return None

    @staticmethod
    def _get_priority(document_type: str) -> int:
        """Get the priority value for a document type.

        Args:
            document_type: The document type string.

        Returns:
            Integer priority value. Defaults to _DEFAULT_PRIORITY
            for unknown types.
        """
        return PRIORITY_ORDER.get(document_type, _DEFAULT_PRIORITY)

    async def _log_conflict(
        self,
        game_id: UUID,
        concept_id: UUID,
        new_source: KGConceptSource,
        previous_source: KGConceptSource,
    ) -> None:
        """Log a conflict resolution event to the audit trail.

        Args:
            game_id: UUID of the owning game.
            concept_id: UUID of the concept.
            new_source: The newly authoritative source.
            previous_source: The previously authoritative source.
        """
        await self._audit_repo.create(
            game_id=game_id,
            event_type="conflict_resolved",
            concept_id=concept_id,
            document_id=new_source.document_id,
            details={
                "concept_id": str(concept_id),
                "new_authoritative_source_id": str(new_source.id),
                "new_document_type": new_source.document_type,
                "new_document_name": new_source.document_name,
                "previous_authoritative_source_id": str(previous_source.id),
                "previous_document_type": previous_source.document_type,
                "previous_document_name": previous_source.document_name,
            },
        )
