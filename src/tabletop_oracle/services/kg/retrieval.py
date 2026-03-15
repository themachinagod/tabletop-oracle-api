"""KG Retrieval Service for semantic search, traversal, and source retrieval.

Internal service consumed by the AI Workflow (EPIC-004). Not exposed as a
public API -- EPIC-004 mediates all user-facing access (PRD-005 Section 3.5).

Provides three retrieval operations:
1. Semantic search: pgvector cosine similarity over concept embeddings.
2. Association traversal: recursive CTE bidirectional graph walk.
3. Source retrieval: full traceability chain for concepts.

All operations enforce per-game isolation (AC-506) and support session-level
expansion filtering (AC-511). Retrieval events are logged to kg_audit_log
(AC-509).

Design reference: docs/architecture/epic-003-knowledge-graph-engine/design.md
    Component Design section 7 (KG Retrieval Service).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sqlalchemy import select, text

from tabletop_oracle.models.knowledge_graph import (
    KGAssociation,
    KGAssociationSource,
    KGAuditLog,
    KGConcept,
    KGConceptSource,
)
from tabletop_oracle.services.kg.search import search_similar_concepts

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from tabletop_oracle.services.kg.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssociationWithContext:
    """An association with its target concept and direction context.

    Attributes:
        association_id: UUID of the association.
        relationship_label: Semantic label (e.g., "modifies", "requires").
        description: Brief description of the relationship.
        related_concept_id: The concept on the other end of the association.
        related_concept_name: Name of the related concept.
        related_concept_type: Semantic type of the related concept.
        direction: Whether this concept is the "source" or "target" end.
    """

    association_id: UUID
    relationship_label: str
    description: str | None
    related_concept_id: UUID
    related_concept_name: str
    related_concept_type: str
    direction: str  # "outgoing" or "incoming"


@dataclass(frozen=True)
class RetrievalResult:
    """A concept returned from semantic search with full context.

    Attributes:
        concept_id: UUID of the matched concept.
        name: Concept name.
        semantic_type: Concept semantic type.
        description: Concept description.
        similarity_score: Cosine similarity score (0-1).
        sources: Source references for this concept.
        associations: Direct associations from/to this concept.
    """

    concept_id: UUID
    name: str
    semantic_type: str
    description: str
    similarity_score: float
    sources: list[SourceReference] = field(default_factory=list)
    associations: list[AssociationWithContext] = field(default_factory=list)


@dataclass(frozen=True)
class SourceReference:
    """Simplified source reference for retrieval results.

    Attributes:
        source_id: UUID of the concept source record.
        document_id: FK to the source document.
        document_name: Cached document name for display.
        document_type: Document type for priority context.
        section_path: Hierarchical section path.
        page_number: Page number in original document.
        source_text: Exact text excerpt (citable foundation).
        is_authoritative: Whether this is the highest-priority source.
        expansion_id: Expansion FK (None for base-game).
    """

    source_id: UUID
    document_id: UUID
    document_name: str
    document_type: str
    section_path: str | None
    page_number: int | None
    source_text: str
    is_authoritative: bool
    expansion_id: UUID | None


@dataclass(frozen=True)
class TraversalNode:
    """A node reached during association traversal.

    Attributes:
        concept_id: UUID of the reached concept.
        concept_name: Name of the reached concept.
        concept_type: Semantic type of the reached concept.
        association_id: UUID of the association used to reach this concept.
        relationship_label: Semantic label on the association.
        direction: Traversal direction ("outgoing" or "incoming").
        depth: Traversal depth (1 = directly connected).
    """

    concept_id: UUID
    concept_name: str
    concept_type: str
    association_id: UUID
    relationship_label: str
    direction: str
    depth: int


@dataclass(frozen=True)
class TraversalResult:
    """Result of association traversal from a starting concept.

    Attributes:
        root_concept_id: UUID of the starting concept.
        root_concept_name: Name of the starting concept.
        nodes: Reachable concepts with traversal metadata.
    """

    root_concept_id: UUID
    root_concept_name: str
    nodes: list[TraversalNode]


@dataclass(frozen=True)
class DocumentContribution:
    """Summary of a document's contribution to the knowledge graph.

    Used for document removal/update operations (PRD-005 Section 4).

    Attributes:
        document_id: UUID of the document.
        game_id: UUID of the game.
        sole_source_concept_ids: Concepts sourced only from this document.
        shared_concept_ids: Concepts that also have sources from other docs.
        association_source_ids: Association sources from this document.
    """

    document_id: UUID
    game_id: UUID
    sole_source_concept_ids: list[UUID]
    shared_concept_ids: list[UUID]
    association_source_ids: list[UUID]


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class KGRetrievalService:
    """Semantic retrieval interface for the AI Workflow.

    Provides semantic search, association traversal, source retrieval, and
    document contribution queries. All operations enforce per-game isolation
    and support session-level expansion filtering.

    Args:
        db: Async database session.
        embedding_service: Service for embedding search queries.
    """

    def __init__(
        self,
        db: AsyncSession,
        embedding_service: EmbeddingService,
    ) -> None:
        self._db = db
        self._embedding_service = embedding_service

    async def semantic_search(
        self,
        game_id: UUID,
        query_text: str,
        active_expansion_ids: list[UUID] | None = None,
        limit: int = 20,
    ) -> list[RetrievalResult]:
        """Find concepts most relevant to a natural-language query.

        Uses pgvector cosine similarity over concept embeddings. Results
        include sources and direct associations for each concept.

        Only canonical concepts are searched (canonical_id IS NULL). Alias
        concepts are excluded -- their canonical concept appears instead.

        Args:
            game_id: Game scope (AC-506).
            query_text: Natural-language search input.
            active_expansion_ids: Active expansion IDs for session filtering
                (AC-511). None means no filtering.
            limit: Maximum number of results.

        Returns:
            Ranked list of RetrievalResult ordered by similarity (descending).
        """
        similarity_results = await search_similar_concepts(
            db=self._db,
            embedding_service=self._embedding_service,
            game_id=game_id,
            query_text=query_text,
            limit=limit,
            active_expansion_ids=active_expansion_ids,
        )

        results: list[RetrievalResult] = []
        for sim in similarity_results:
            sources = await self._get_sources_for_concept(sim.concept_id, active_expansion_ids)
            associations = await self._get_direct_associations(
                sim.concept_id, game_id, active_expansion_ids
            )
            results.append(
                RetrievalResult(
                    concept_id=sim.concept_id,
                    name=sim.name,
                    semantic_type=sim.semantic_type,
                    description=sim.description,
                    similarity_score=sim.similarity_score,
                    sources=sources,
                    associations=associations,
                )
            )

        await self._log_retrieval_event(
            game_id=game_id,
            query_text=query_text,
            results_count=len(results),
            expansion_filter_active=active_expansion_ids is not None,
        )

        return results

    async def traverse_associations(
        self,
        game_id: UUID,
        concept_id: UUID,
        max_depth: int = 3,
        active_expansion_ids: list[UUID] | None = None,
    ) -> TraversalResult:
        """Traverse associations bidirectionally from a concept.

        Uses a recursive CTE to walk the association graph up to max_depth
        hops. Traversal is bidirectional: follows associations where the
        concept is either source or target.

        Args:
            game_id: Game scope (AC-506).
            concept_id: Starting concept UUID.
            max_depth: Maximum traversal depth (default 3).
            active_expansion_ids: Active expansion IDs for session filtering.

        Returns:
            TraversalResult with root concept and all reachable nodes.
        """
        # Get root concept name for the result
        root_stmt = select(KGConcept.name).where(
            KGConcept.id == concept_id,
            KGConcept.game_id == game_id,
        )
        root_result = await self._db.execute(root_stmt)
        root_name = root_result.scalar_one_or_none()

        if root_name is None:
            return TraversalResult(
                root_concept_id=concept_id,
                root_concept_name="",
                nodes=[],
            )

        # Build the recursive CTE for bidirectional traversal.
        # The CTE walks outgoing and incoming associations, tracking visited
        # concepts to avoid cycles, up to max_depth.
        #
        # We use raw SQL for the recursive CTE because SQLAlchemy's CTE
        # support for recursive self-joins with bidirectional traversal and
        # cycle detection is awkward to express cleanly.
        expansion_filter_clause = ""
        params: dict[str, object] = {
            "concept_id": str(concept_id),
            "game_id": str(game_id),
            "max_depth": max_depth,
        }

        if active_expansion_ids is not None:
            if active_expansion_ids:
                expansion_filter_clause = """
                    AND EXISTS (
                        SELECT 1 FROM kg_concept_sources s
                        WHERE s.concept_id = related.id
                        AND (s.expansion_id IS NULL
                             OR s.expansion_id = ANY(:active_expansion_ids))
                    )
                """
                params["active_expansion_ids"] = [str(eid) for eid in active_expansion_ids]
            else:
                # Empty list = base game only
                expansion_filter_clause = """
                    AND EXISTS (
                        SELECT 1 FROM kg_concept_sources s
                        WHERE s.concept_id = related.id
                        AND s.expansion_id IS NULL
                    )
                """

        cte_sql = text(f"""
            WITH RECURSIVE traversal AS (
                -- Anchor: direct outgoing associations
                SELECT
                    a.id AS association_id,
                    a.relationship_label,
                    a.target_concept_id AS related_id,
                    'outgoing' AS direction,
                    1 AS depth,
                    ARRAY[a.target_concept_id] AS visited
                FROM kg_associations a
                JOIN kg_concepts related ON related.id = a.target_concept_id
                WHERE a.source_concept_id = :concept_id
                  AND a.game_id = :game_id
                  AND related.canonical_id IS NULL
                  {expansion_filter_clause}

                UNION ALL

                -- Anchor: direct incoming associations
                SELECT
                    a.id AS association_id,
                    a.relationship_label,
                    a.source_concept_id AS related_id,
                    'incoming' AS direction,
                    1 AS depth,
                    ARRAY[a.source_concept_id] AS visited
                FROM kg_associations a
                JOIN kg_concepts related ON related.id = a.source_concept_id
                WHERE a.target_concept_id = :concept_id
                  AND a.game_id = :game_id
                  AND related.canonical_id IS NULL
                  {expansion_filter_clause}

                UNION ALL

                -- Recursive: walk further from already-discovered concepts
                SELECT
                    a.id AS association_id,
                    a.relationship_label,
                    CASE
                        WHEN a.source_concept_id = t.related_id
                            THEN a.target_concept_id
                        ELSE a.source_concept_id
                    END AS related_id,
                    CASE
                        WHEN a.source_concept_id = t.related_id
                            THEN 'outgoing'
                        ELSE 'incoming'
                    END AS direction,
                    t.depth + 1 AS depth,
                    t.visited || CASE
                        WHEN a.source_concept_id = t.related_id
                            THEN a.target_concept_id
                        ELSE a.source_concept_id
                    END AS visited
                FROM kg_associations a
                JOIN traversal t ON (
                    a.source_concept_id = t.related_id
                    OR a.target_concept_id = t.related_id
                )
                JOIN kg_concepts related ON related.id = CASE
                    WHEN a.source_concept_id = t.related_id
                        THEN a.target_concept_id
                    ELSE a.source_concept_id
                END
                WHERE t.depth < :max_depth
                  AND a.game_id = :game_id
                  AND related.canonical_id IS NULL
                  AND NOT (CASE
                      WHEN a.source_concept_id = t.related_id
                          THEN a.target_concept_id
                      ELSE a.source_concept_id
                  END = ANY(t.visited))
                  AND CASE
                      WHEN a.source_concept_id = t.related_id
                          THEN a.target_concept_id
                      ELSE a.source_concept_id
                  END != :concept_id
                  {expansion_filter_clause}
            )
            SELECT DISTINCT ON (t.related_id)
                t.association_id,
                t.relationship_label,
                t.related_id,
                t.direction,
                t.depth,
                c.name AS concept_name,
                c.semantic_type AS concept_type
            FROM traversal t
            JOIN kg_concepts c ON c.id = t.related_id
            ORDER BY t.related_id, t.depth
        """)

        result = await self._db.execute(cte_sql, params)
        rows = result.all()

        nodes = [
            TraversalNode(
                concept_id=row.related_id,
                concept_name=row.concept_name,
                concept_type=row.concept_type,
                association_id=row.association_id,
                relationship_label=row.relationship_label,
                direction=row.direction,
                depth=row.depth,
            )
            for row in rows
        ]

        return TraversalResult(
            root_concept_id=concept_id,
            root_concept_name=root_name,
            nodes=nodes,
        )

    async def get_concept_sources(
        self,
        concept_id: UUID,
        active_expansion_ids: list[UUID] | None = None,
    ) -> list[SourceReference]:
        """Get full source traceability for a concept.

        Returns all source references ordered by authoritative status
        (authoritative first).

        Args:
            concept_id: UUID of the concept.
            active_expansion_ids: Active expansion IDs for session filtering.

        Returns:
            List of SourceReference ordered by is_authoritative DESC.
        """
        return await self._get_sources_for_concept(concept_id, active_expansion_ids)

    async def get_document_contribution(
        self,
        document_id: UUID,
        game_id: UUID,
    ) -> DocumentContribution:
        """Get all concepts and associations contributed by a document.

        Identifies concepts solely sourced from this document (would be
        orphaned on removal) vs. concepts with additional sources.

        Args:
            document_id: UUID of the document.
            game_id: UUID of the game (AC-506).

        Returns:
            DocumentContribution summarising the document's KG impact.
        """
        # Find all concept sources for this document
        concept_source_stmt = (
            select(KGConceptSource.concept_id)
            .where(KGConceptSource.document_id == document_id)
            .join(KGConcept, KGConcept.id == KGConceptSource.concept_id)
            .where(KGConcept.game_id == game_id)
        )
        result = await self._db.execute(concept_source_stmt)
        contributed_concept_ids = [row[0] for row in result.all()]

        sole_source_ids: list[UUID] = []
        shared_ids: list[UUID] = []

        for cid in set(contributed_concept_ids):
            # Count sources from OTHER documents
            other_sources_stmt = (
                select(KGConceptSource.id)
                .where(
                    KGConceptSource.concept_id == cid,
                    KGConceptSource.document_id != document_id,
                )
                .limit(1)
            )
            other_result = await self._db.execute(other_sources_stmt)
            if other_result.scalar_one_or_none() is None:
                sole_source_ids.append(cid)
            else:
                shared_ids.append(cid)

        # Find association sources from this document
        assoc_source_stmt = (
            select(KGAssociationSource.id)
            .where(KGAssociationSource.document_id == document_id)
            .join(
                KGAssociation,
                KGAssociation.id == KGAssociationSource.association_id,
            )
            .where(KGAssociation.game_id == game_id)
        )
        assoc_result = await self._db.execute(assoc_source_stmt)
        assoc_source_ids = [row[0] for row in assoc_result.all()]

        return DocumentContribution(
            document_id=document_id,
            game_id=game_id,
            sole_source_concept_ids=sole_source_ids,
            shared_concept_ids=shared_ids,
            association_source_ids=assoc_source_ids,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _get_sources_for_concept(
        self,
        concept_id: UUID,
        active_expansion_ids: list[UUID] | None,
    ) -> list[SourceReference]:
        """Fetch source references for a concept with optional expansion filter.

        Args:
            concept_id: UUID of the concept.
            active_expansion_ids: Expansion filter. None = no filtering.

        Returns:
            List of SourceReference ordered by is_authoritative DESC.
        """
        stmt = select(KGConceptSource).where(KGConceptSource.concept_id == concept_id)

        if active_expansion_ids is not None:
            if active_expansion_ids:
                stmt = stmt.where(
                    KGConceptSource.expansion_id.is_(None)
                    | KGConceptSource.expansion_id.in_(active_expansion_ids)
                )
            else:
                # Empty list = base game only
                stmt = stmt.where(KGConceptSource.expansion_id.is_(None))

        stmt = stmt.order_by(KGConceptSource.is_authoritative.desc())

        result = await self._db.execute(stmt)
        rows = result.scalars().all()

        return [
            SourceReference(
                source_id=src.id,
                document_id=src.document_id,
                document_name=src.document_name,
                document_type=src.document_type,
                section_path=src.section_path,
                page_number=src.page_number,
                source_text=src.source_text,
                is_authoritative=src.is_authoritative,
                expansion_id=src.expansion_id,
            )
            for src in rows
        ]

    async def _get_direct_associations(
        self,
        concept_id: UUID,
        game_id: UUID,
        active_expansion_ids: list[UUID] | None,
    ) -> list[AssociationWithContext]:
        """Fetch direct associations for a concept (both directions).

        Args:
            concept_id: UUID of the concept.
            game_id: Game scope.
            active_expansion_ids: Expansion filter.

        Returns:
            List of direct associations with context.
        """
        # Outgoing associations (concept is source)
        outgoing_stmt = (
            select(
                KGAssociation.id,
                KGAssociation.relationship_label,
                KGAssociation.description,
                KGConcept.id.label("related_id"),
                KGConcept.name.label("related_name"),
                KGConcept.semantic_type.label("related_type"),
            )
            .join(KGConcept, KGConcept.id == KGAssociation.target_concept_id)
            .where(
                KGAssociation.source_concept_id == concept_id,
                KGAssociation.game_id == game_id,
                KGConcept.canonical_id.is_(None),
            )
        )

        # Incoming associations (concept is target)
        incoming_stmt = (
            select(
                KGAssociation.id,
                KGAssociation.relationship_label,
                KGAssociation.description,
                KGConcept.id.label("related_id"),
                KGConcept.name.label("related_name"),
                KGConcept.semantic_type.label("related_type"),
            )
            .join(KGConcept, KGConcept.id == KGAssociation.source_concept_id)
            .where(
                KGAssociation.target_concept_id == concept_id,
                KGAssociation.game_id == game_id,
                KGConcept.canonical_id.is_(None),
            )
        )

        # Apply expansion filter if needed
        if active_expansion_ids is not None:
            expansion_subquery = (
                select(KGConceptSource.concept_id)
                .where(KGConceptSource.concept_id == KGConcept.id)
                .correlate(KGConcept)
            )
            if active_expansion_ids:
                expansion_subquery = expansion_subquery.where(
                    KGConceptSource.expansion_id.is_(None)
                    | KGConceptSource.expansion_id.in_(active_expansion_ids)
                )
            else:
                expansion_subquery = expansion_subquery.where(
                    KGConceptSource.expansion_id.is_(None)
                )
            outgoing_stmt = outgoing_stmt.where(expansion_subquery.exists())
            incoming_stmt = incoming_stmt.where(expansion_subquery.exists())

        associations: list[AssociationWithContext] = []

        out_result = await self._db.execute(outgoing_stmt)
        for row in out_result.all():
            associations.append(
                AssociationWithContext(
                    association_id=row.id,
                    relationship_label=row.relationship_label,
                    description=row.description,
                    related_concept_id=row.related_id,
                    related_concept_name=row.related_name,
                    related_concept_type=row.related_type,
                    direction="outgoing",
                )
            )

        in_result = await self._db.execute(incoming_stmt)
        for row in in_result.all():
            associations.append(
                AssociationWithContext(
                    association_id=row.id,
                    relationship_label=row.relationship_label,
                    description=row.description,
                    related_concept_id=row.related_id,
                    related_concept_name=row.related_name,
                    related_concept_type=row.related_type,
                    direction="incoming",
                )
            )

        return associations

    async def _log_retrieval_event(
        self,
        game_id: UUID,
        query_text: str,
        results_count: int,
        expansion_filter_active: bool,
    ) -> None:
        """Log a retrieval event to the kg_audit_log table.

        Args:
            game_id: Game scope.
            query_text: The search query.
            results_count: Number of results returned.
            expansion_filter_active: Whether expansion filtering was used.
        """
        audit_entry = KGAuditLog(
            game_id=game_id,
            event_type="retrieval",
            details={
                "query_text": query_text,
                "results_count": results_count,
                "expansion_filter_active": expansion_filter_active,
            },
        )
        self._db.add(audit_entry)
        await self._db.flush()
