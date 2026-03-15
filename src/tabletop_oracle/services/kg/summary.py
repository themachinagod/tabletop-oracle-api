"""KG summary service for curator-facing aggregate metrics.

Provides aggregate KG statistics, per-document contribution queries, and
reprocess triggering. Consumed by the curator KG summary API router.

Design reference: docs/architecture/epic-003-knowledge-graph-engine/design.md
    API Contracts section — Curator KG Summary endpoints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import distinct, func, select

from tabletop_oracle.models.knowledge_graph import (
    KGAssociation,
    KGAssociationSource,
    KGAuditLog,
    KGConcept,
    KGConceptSource,
)

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemanticTypeStat:
    """Count of concepts by semantic type.

    Attributes:
        semantic_type: The semantic type label.
        count: Number of canonical concepts with this type.
    """

    semantic_type: str
    count: int


@dataclass(frozen=True)
class DocumentSummaryData:
    """Per-document contribution summary.

    Attributes:
        document_id: UUID of the contributing document.
        document_name: Display name of the document.
        document_type: Document type classification.
        concepts_contributed: Number of concepts sourced from this document.
        associations_contributed: Number of associations sourced from this document.
    """

    document_id: UUID
    document_name: str
    document_type: str
    concepts_contributed: int
    associations_contributed: int


@dataclass(frozen=True)
class KGSummaryData:
    """Aggregate KG metrics for a game.

    Attributes:
        game_id: UUID of the game.
        concept_count: Total canonical concept count.
        association_count: Total association count.
        document_count: Number of distinct contributing documents.
        semantic_types: Concept counts grouped by semantic type.
        documents: Per-document contribution summaries.
        last_updated_at: Most recent KG modification timestamp.
    """

    game_id: UUID
    concept_count: int
    association_count: int
    document_count: int
    semantic_types: list[SemanticTypeStat]
    documents: list[DocumentSummaryData]
    last_updated_at: datetime | None


@dataclass(frozen=True)
class DocumentContributionData:
    """Per-document contribution detail for paginated listing.

    Attributes:
        document_id: UUID of the document.
        document_name: Display name of the document.
        document_type: Document type classification.
        expansion_id: Expansion UUID if the document belongs to an expansion.
        concepts_contributed: Number of concepts sourced from this document.
        associations_contributed: Number of associations sourced from this document.
        conflicts_detected: Number of conflict events for this document.
        last_processed_at: Most recent processing timestamp for this document.
    """

    document_id: UUID
    document_name: str
    document_type: str
    expansion_id: UUID | None
    concepts_contributed: int
    associations_contributed: int
    conflicts_detected: int
    last_processed_at: datetime | None


class KGSummaryService:
    """Curator-facing KG summary queries and reprocess triggering.

    Provides aggregate metrics, per-document contribution details, and
    the ability to trigger a full KG rebuild for a game.

    Args:
        db: Async database session.
    """

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get_summary(self, game_id: UUID) -> KGSummaryData:
        """Get aggregate KG metrics for a game (AC-510).

        Counts canonical concepts (canonical_id IS NULL), associations,
        contributing documents, and groups concepts by semantic type.

        Args:
            game_id: UUID of the game.

        Returns:
            Aggregate KG statistics for the game.
        """
        # Count canonical concepts
        concept_count_stmt = select(func.count(KGConcept.id)).where(
            KGConcept.game_id == game_id,
            KGConcept.canonical_id.is_(None),
        )
        concept_count = (await self._db.execute(concept_count_stmt)).scalar_one()

        # Count associations
        assoc_count_stmt = select(func.count(KGAssociation.id)).where(
            KGAssociation.game_id == game_id,
        )
        association_count = (await self._db.execute(assoc_count_stmt)).scalar_one()

        # Count distinct contributing documents
        doc_count_stmt = select(func.count(distinct(KGConceptSource.document_id))).where(
            KGConceptSource.concept_id.in_(
                select(KGConcept.id).where(
                    KGConcept.game_id == game_id,
                    KGConcept.canonical_id.is_(None),
                )
            )
        )
        document_count = (await self._db.execute(doc_count_stmt)).scalar_one()

        # Semantic type breakdown
        type_stmt = (
            select(
                KGConcept.semantic_type,
                func.count(KGConcept.id).label("cnt"),
            )
            .where(
                KGConcept.game_id == game_id,
                KGConcept.canonical_id.is_(None),
            )
            .group_by(KGConcept.semantic_type)
            .order_by(func.count(KGConcept.id).desc())
        )
        type_rows = (await self._db.execute(type_stmt)).all()
        semantic_types = [
            SemanticTypeStat(semantic_type=row.semantic_type, count=row.cnt) for row in type_rows
        ]

        # Per-document contribution summary
        documents = await self._get_document_summaries(game_id)

        # Last updated timestamp (most recent concept or association update)
        last_updated_at = await self._get_last_updated(game_id)

        return KGSummaryData(
            game_id=game_id,
            concept_count=concept_count,
            association_count=association_count,
            document_count=document_count,
            semantic_types=semantic_types,
            documents=documents,
            last_updated_at=last_updated_at,
        )

    async def get_document_contributions(
        self,
        game_id: UUID,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[DocumentContributionData], int]:
        """Get per-document contribution detail with pagination.

        Returns documents that have contributed concepts to the KG for
        the given game, with counts and conflict information.

        Args:
            game_id: UUID of the game.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            Tuple of (document contributions, total count).
        """
        # Subquery: documents contributing concepts to this game's KG
        contributing_docs_sq = (
            select(
                KGConceptSource.document_id,
                KGConceptSource.document_name,
                KGConceptSource.document_type,
                KGConceptSource.expansion_id,
            )
            .join(KGConcept, KGConcept.id == KGConceptSource.concept_id)
            .where(
                KGConcept.game_id == game_id,
                KGConcept.canonical_id.is_(None),
            )
            .distinct(KGConceptSource.document_id)
            .subquery()
        )

        # Count total distinct documents
        total_count_stmt = select(func.count()).select_from(contributing_docs_sq)
        total_count = (await self._db.execute(total_count_stmt)).scalar_one()

        # Get distinct document IDs with pagination
        doc_ids_stmt = (
            select(contributing_docs_sq.c.document_id)
            .order_by(contributing_docs_sq.c.document_name)
            .limit(limit)
            .offset(offset)
        )
        doc_id_rows = (await self._db.execute(doc_ids_stmt)).all()

        contributions: list[DocumentContributionData] = []
        for row in doc_id_rows:
            doc_id = row.document_id
            contribution = await self._get_single_document_contribution(game_id, doc_id)
            if contribution is not None:
                contributions.append(contribution)

        return contributions, total_count

    async def count_documents_for_reprocess(self, game_id: UUID) -> int:
        """Count documents that have contributed to the KG for a game.

        Used to determine how many documents would be reprocessed.

        Args:
            game_id: UUID of the game.

        Returns:
            Number of distinct contributing documents.
        """
        stmt = select(func.count(distinct(KGConceptSource.document_id))).where(
            KGConceptSource.concept_id.in_(
                select(KGConcept.id).where(
                    KGConcept.game_id == game_id,
                    KGConcept.canonical_id.is_(None),
                )
            )
        )
        return (await self._db.execute(stmt)).scalar_one()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _get_document_summaries(self, game_id: UUID) -> list[DocumentSummaryData]:
        """Get per-document contribution summaries for the summary endpoint.

        Args:
            game_id: UUID of the game.

        Returns:
            List of DocumentSummaryData sorted by concepts contributed descending.
        """
        # Concept contributions per document
        concept_stmt = (
            select(
                KGConceptSource.document_id,
                KGConceptSource.document_name,
                KGConceptSource.document_type,
                func.count(distinct(KGConceptSource.concept_id)).label("concept_count"),
            )
            .join(KGConcept, KGConcept.id == KGConceptSource.concept_id)
            .where(
                KGConcept.game_id == game_id,
                KGConcept.canonical_id.is_(None),
            )
            .group_by(
                KGConceptSource.document_id,
                KGConceptSource.document_name,
                KGConceptSource.document_type,
            )
            .order_by(func.count(distinct(KGConceptSource.concept_id)).desc())
        )
        concept_rows = (await self._db.execute(concept_stmt)).all()

        summaries: list[DocumentSummaryData] = []
        for row in concept_rows:
            # Count association contributions for this document
            assoc_stmt = (
                select(func.count(distinct(KGAssociationSource.association_id)))
                .join(
                    KGAssociation,
                    KGAssociation.id == KGAssociationSource.association_id,
                )
                .where(
                    KGAssociationSource.document_id == row.document_id,
                    KGAssociation.game_id == game_id,
                )
            )
            assoc_count = (await self._db.execute(assoc_stmt)).scalar_one()

            summaries.append(
                DocumentSummaryData(
                    document_id=row.document_id,
                    document_name=row.document_name,
                    document_type=row.document_type,
                    concepts_contributed=row.concept_count,
                    associations_contributed=assoc_count,
                )
            )

        return summaries

    async def _get_last_updated(self, game_id: UUID) -> datetime | None:
        """Get the most recent KG modification timestamp for a game.

        Checks both concept and association updated_at timestamps.

        Args:
            game_id: UUID of the game.

        Returns:
            Most recent timestamp, or None if no KG data exists.
        """
        concept_max_stmt = select(func.max(KGConcept.updated_at)).where(
            KGConcept.game_id == game_id,
        )
        concept_max = (await self._db.execute(concept_max_stmt)).scalar_one_or_none()

        assoc_max_stmt = select(func.max(KGAssociation.updated_at)).where(
            KGAssociation.game_id == game_id,
        )
        assoc_max = (await self._db.execute(assoc_max_stmt)).scalar_one_or_none()

        if concept_max is None and assoc_max is None:
            return None
        if concept_max is None:
            return assoc_max
        if assoc_max is None:
            return concept_max
        return max(concept_max, assoc_max)

    async def _get_single_document_contribution(
        self,
        game_id: UUID,
        document_id: UUID,
    ) -> DocumentContributionData | None:
        """Get contribution detail for a single document.

        Args:
            game_id: UUID of the game.
            document_id: UUID of the document.

        Returns:
            Contribution data, or None if the document has no contributions.
        """
        # Get document metadata from concept sources
        meta_stmt = (
            select(
                KGConceptSource.document_name,
                KGConceptSource.document_type,
                KGConceptSource.expansion_id,
            )
            .join(KGConcept, KGConcept.id == KGConceptSource.concept_id)
            .where(
                KGConceptSource.document_id == document_id,
                KGConcept.game_id == game_id,
                KGConcept.canonical_id.is_(None),
            )
            .limit(1)
        )
        meta_row = (await self._db.execute(meta_stmt)).first()
        if meta_row is None:
            return None

        # Count concept contributions
        concept_count_stmt = (
            select(func.count(distinct(KGConceptSource.concept_id)))
            .join(KGConcept, KGConcept.id == KGConceptSource.concept_id)
            .where(
                KGConceptSource.document_id == document_id,
                KGConcept.game_id == game_id,
                KGConcept.canonical_id.is_(None),
            )
        )
        concepts_contributed = (await self._db.execute(concept_count_stmt)).scalar_one()

        # Count association contributions
        assoc_count_stmt = (
            select(func.count(distinct(KGAssociationSource.association_id)))
            .join(
                KGAssociation,
                KGAssociation.id == KGAssociationSource.association_id,
            )
            .where(
                KGAssociationSource.document_id == document_id,
                KGAssociation.game_id == game_id,
            )
        )
        associations_contributed = (await self._db.execute(assoc_count_stmt)).scalar_one()

        # Count conflict events from audit log
        conflict_count_stmt = select(func.count(KGAuditLog.id)).where(
            KGAuditLog.game_id == game_id,
            KGAuditLog.document_id == document_id,
            KGAuditLog.event_type == "conflict_detected",
        )
        conflicts_detected = (await self._db.execute(conflict_count_stmt)).scalar_one()

        # Last processed timestamp from concept sources
        last_processed_stmt = select(func.max(KGConceptSource.created_at)).where(
            KGConceptSource.document_id == document_id,
            KGConceptSource.concept_id.in_(
                select(KGConcept.id).where(
                    KGConcept.game_id == game_id,
                )
            ),
        )
        last_processed_at = (await self._db.execute(last_processed_stmt)).scalar_one_or_none()

        return DocumentContributionData(
            document_id=document_id,
            document_name=meta_row.document_name,
            document_type=meta_row.document_type,
            expansion_id=meta_row.expansion_id,
            concepts_contributed=concepts_contributed,
            associations_contributed=associations_contributed,
            conflicts_detected=conflicts_detected,
            last_processed_at=last_processed_at,
        )
