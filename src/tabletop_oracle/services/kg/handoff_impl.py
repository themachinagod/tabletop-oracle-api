"""Real KG handoff service implementation.

Replaces the EPIC-002 no-op stub. Orchestrates the full KG construction
pipeline: extract concepts, discover associations, integrate into graph,
resolve conflicts, generate embeddings, and log all events.

Design reference: docs/architecture/epic-003-knowledge-graph-engine/design.md
    Component Design section 1 (KG Handoff Service Implementation).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from tabletop_oracle.services.ingestion.handoff import KGIngestionError
from tabletop_oracle.services.kg.associations import ConceptSummary
from tabletop_oracle.services.kg.extraction import GameContext
from tabletop_oracle.services.model.token_usage import TokenAttribution

if TYPE_CHECKING:
    from uuid import UUID

    from tabletop_oracle.models.knowledge_graph import KGConcept
    from tabletop_oracle.services.ingestion.events import EventPublisher
    from tabletop_oracle.services.ingestion.handoff import KGHandoffPayload
    from tabletop_oracle.services.ingestion.models import Chunk
    from tabletop_oracle.services.kg.associations import (
        AssociationDiscoveryService,
        DiscoveredAssociation,
    )
    from tabletop_oracle.services.kg.audit import KGAuditService
    from tabletop_oracle.services.kg.conflicts import ConflictResolutionService
    from tabletop_oracle.services.kg.embeddings import EmbeddingService
    from tabletop_oracle.services.kg.extraction import (
        ConceptExtractionService,
        ExtractedConcept,
    )
    from tabletop_oracle.services.kg.integration import GraphIntegrationService

logger = logging.getLogger(__name__)

#: Async callable returning game context for a given game_id.
GameContextProvider = Callable[["UUID"], Awaitable[GameContext]]

#: Async callable returning token attribution for a given document_id.
AttributionProvider = Callable[["UUID"], Awaitable[TokenAttribution]]

#: Async callable returning existing concept summaries for a game.
ExistingConceptsProvider = Callable[["UUID"], Awaitable[list[ConceptSummary]]]


class KGHandoffServiceImpl:
    """Real implementation of the KG handoff interface.

    Receives processed document chunks from the ingestion pipeline and
    orchestrates KG construction: extraction, association discovery,
    graph integration, conflict resolution, and embedding generation.

    Satisfies the ``KGHandoffService`` protocol defined in
    ``tabletop_oracle.services.ingestion.handoff``.

    Args:
        extraction_service: Extracts concepts from chunks via LLM.
        association_service: Discovers associations between concepts via LLM.
        integration_service: Merges new concepts into the existing graph.
        conflict_service: Resolves conflicts using document type priority.
        embedding_service: Generates vector embeddings for concepts.
        audit_service: Logs all KG construction events.
        game_context_provider: Callable that returns GameContext for a game_id.
        attribution_provider: Callable that returns TokenAttribution for a
            document_id.
        event_publisher: Optional SSE event publisher for construction stage
            events.
        existing_concepts_provider: Optional callable returning existing concept
            summaries for a game. Used for cross-document association discovery.
    """

    def __init__(
        self,
        extraction_service: ConceptExtractionService,
        association_service: AssociationDiscoveryService,
        integration_service: GraphIntegrationService,
        conflict_service: ConflictResolutionService,
        embedding_service: EmbeddingService,
        audit_service: KGAuditService,
        *,
        game_context_provider: GameContextProvider,
        attribution_provider: AttributionProvider,
        event_publisher: EventPublisher | None = None,
        existing_concepts_provider: ExistingConceptsProvider | None = None,
    ) -> None:
        self._extraction = extraction_service
        self._associations = association_service
        self._integration = integration_service
        self._conflicts = conflict_service
        self._embeddings = embedding_service
        self._audit = audit_service
        self._game_context_provider = game_context_provider
        self._attribution_provider = attribution_provider
        self._event_publisher = event_publisher
        self._existing_concepts_provider = existing_concepts_provider

    async def ingest_document(self, payload: KGHandoffPayload) -> None:
        """Process chunks into KG concepts and associations.

        Pipeline:
        1. If replaces_version_id is set, remove old version's contributions.
        2. Extract concepts from each chunk via LLM.
        3. Discover associations between extracted concepts via LLM.
        4. Integrate into existing game graph (merge duplicates, cross-doc).
        5. Resolve conflicts using document type priority.
        6. Generate embeddings for new/updated concepts.
        7. Log all construction events.

        Args:
            payload: Complete handoff payload with document metadata and chunks.

        Raises:
            KGIngestionError: On failure to ingest.
        """
        try:
            await self._run_pipeline(payload)
        except KGIngestionError:
            raise
        except Exception as exc:
            raise KGIngestionError(
                f"KG ingestion failed for document {payload.document_id}: {exc}",
                document_id=payload.document_id,
            ) from exc

    async def remove_document(self, document_id: UUID, game_id: UUID) -> None:
        """Remove a document's contribution from the KG.

        Removes concepts solely sourced from this document. For concepts
        with multiple sources, removes this document's source reference
        but retains the concept. Cleans up orphaned associations.
        Logs the removal event.

        Args:
            document_id: UUID of the document to remove.
            game_id: UUID of the owning game.

        Raises:
            KGIngestionError: On failure to remove.
        """
        try:
            self._emit_event(document_id, "kg.removing", {"stage": "removing"})

            removal_result = await self._integration.remove_document_contributions(
                game_id=game_id,
                document_id=document_id,
            )

            await self._audit.log_document_removal(
                game_id=game_id,
                document_id=document_id,
                concepts_removed=removal_result.concepts_removed,
                concepts_updated=removal_result.concepts_updated,
            )

            self._emit_event(
                document_id,
                "kg.removed",
                {
                    "concepts_removed": removal_result.concepts_removed,
                    "concepts_updated": removal_result.concepts_updated,
                },
            )

            logger.info(
                "Document removed from KG: document_id=%s game_id=%s "
                "concepts_removed=%d concepts_updated=%d",
                document_id,
                game_id,
                removal_result.concepts_removed,
                removal_result.concepts_updated,
            )

        except KGIngestionError:
            raise
        except Exception as exc:
            raise KGIngestionError(
                f"KG removal failed for document {document_id}: {exc}",
                document_id=document_id,
            ) from exc

    async def _run_pipeline(self, payload: KGHandoffPayload) -> None:
        """Execute the full KG construction pipeline.

        Args:
            payload: Handoff payload with document metadata and chunks.
        """
        # Step 1: Version replacement -- remove old version first
        if payload.replaces_version_id is not None:
            logger.info(
                "Version replacement: removing contributions from version %s",
                payload.replaces_version_id,
            )
            await self.remove_document(payload.document_id, payload.game_id)

        if not payload.chunks:
            logger.info(
                "No chunks to process for document %s, skipping KG pipeline",
                payload.document_id,
            )
            return

        game_context = await self._game_context_provider(payload.game_id)
        attribution = await self._attribution_provider(payload.document_id)

        # Step 2: Extract concepts
        self._emit_event(
            payload.document_id,
            "kg.extracting",
            {"stage": "extracting", "chunk_count": len(payload.chunks)},
        )
        all_concepts = await self._extract_all_concepts(payload.chunks, game_context, attribution)

        if not all_concepts:
            logger.info(
                "No concepts extracted from document %s",
                payload.document_id,
            )
            return

        # Step 3: Discover associations
        self._emit_event(
            payload.document_id,
            "kg.discovering_associations",
            {
                "stage": "discovering_associations",
                "concept_count": len(all_concepts),
            },
        )
        existing_concepts = await self._get_existing_concepts(payload.game_id)
        all_associations = await self._discover_all_associations(
            payload.chunks, all_concepts, existing_concepts, attribution
        )

        # Step 4: Integrate into graph
        self._emit_event(
            payload.document_id,
            "kg.integrating",
            {
                "stage": "integrating",
                "concept_count": len(all_concepts),
                "association_count": len(all_associations),
            },
        )
        integrated_concepts = await self._integration.integrate_concepts(
            game_id=payload.game_id,
            document_id=payload.document_id,
            version_id=payload.version_id,
            document_type=payload.document_type,
            document_name=str(payload.document_id),
            expansion_id=payload.expansion_id,
            extracted_concepts=all_concepts,
        )

        concept_map = self._integration.build_concept_map(integrated_concepts)
        integrated_associations = await self._integration.integrate_associations(
            game_id=payload.game_id,
            document_id=payload.document_id,
            associations=all_associations,
            concept_map=concept_map,
            expansion_id=payload.expansion_id,
        )

        # Step 5: Resolve conflicts
        self._emit_event(
            payload.document_id,
            "kg.resolving_conflicts",
            {"stage": "resolving_conflicts"},
        )
        await self._resolve_all_conflicts(
            game_id=payload.game_id,
            integrated_concepts=integrated_concepts,
        )

        # Step 6: Generate embeddings
        self._emit_event(
            payload.document_id,
            "kg.embedding",
            {"stage": "embedding", "concept_count": len(integrated_concepts)},
        )
        await self._embed_concepts(integrated_concepts)

        # Step 7: Audit logging
        await self._audit.log_extraction(
            game_id=payload.game_id,
            document_id=payload.document_id,
            concept_count=len(all_concepts),
            chunk_count=len(payload.chunks),
        )
        await self._audit.log_association_discovery(
            game_id=payload.game_id,
            document_id=payload.document_id,
            association_count=len(all_associations),
            cross_document_count=self._count_cross_document(all_associations, all_concepts),
        )

        self._emit_event(
            payload.document_id,
            "kg.complete",
            {
                "stage": "complete",
                "concept_count": len(integrated_concepts),
                "association_count": len(integrated_associations),
            },
        )

        logger.info(
            "KG pipeline complete: document_id=%s concepts=%d associations=%d",
            payload.document_id,
            len(integrated_concepts),
            len(integrated_associations),
        )

    async def _extract_all_concepts(
        self,
        chunks: tuple[Chunk, ...],
        game_context: GameContext,
        attribution: TokenAttribution,
    ) -> list[ExtractedConcept]:
        """Extract concepts from all chunks with incremental dedup awareness.

        Processes chunks sequentially, feeding accumulated concept names
        to subsequent extractions for deduplication (AC-507 incremental
        enrichment).

        Args:
            chunks: Document chunks to extract from.
            game_context: Game metadata for extraction prompts.
            attribution: Token usage attribution.

        Returns:
            Flat list of all extracted concepts across all chunks.
        """
        all_concepts: list[ExtractedConcept] = []
        accumulated_names: list[str] = []

        for chunk in chunks:
            try:
                concepts = await self._extraction.extract_from_chunk(
                    chunk=chunk,
                    game_context=game_context,
                    attribution=attribution,
                    existing_concepts=(accumulated_names if accumulated_names else None),
                )
                all_concepts.extend(concepts)
                accumulated_names.extend(c.name for c in concepts)
            except Exception:
                logger.warning(
                    "Concept extraction failed for chunk %d, skipping",
                    chunk.chunk_index,
                    exc_info=True,
                )

        return all_concepts

    async def _discover_all_associations(
        self,
        chunks: tuple[Chunk, ...],
        all_concepts: list[ExtractedConcept],
        existing_concepts: list[ConceptSummary],
        attribution: TokenAttribution,
    ) -> list[DiscoveredAssociation]:
        """Discover associations for each chunk's concepts.

        Groups concepts by their source chunk and runs association
        discovery per chunk, providing existing KG concepts for
        cross-document linking.

        Args:
            chunks: Document chunks.
            all_concepts: All extracted concepts (grouped by chunk).
            existing_concepts: Existing KG concept summaries.
            attribution: Token usage attribution.

        Returns:
            Flat list of all discovered associations.
        """
        concepts_by_chunk: dict[int, list[ExtractedConcept]] = defaultdict(list)
        for concept in all_concepts:
            for i, chunk in enumerate(chunks):
                chunk_id = self._extraction._resolve_chunk_id(chunk)
                if concept.source_chunk_id == chunk_id:
                    concepts_by_chunk[i].append(concept)
                    break

        all_associations: list[DiscoveredAssociation] = []

        for i, chunk in enumerate(chunks):
            chunk_concepts = concepts_by_chunk.get(i, [])
            if not chunk_concepts:
                continue

            try:
                associations = await self._associations.discover_associations(
                    new_concepts=chunk_concepts,
                    existing_concepts=existing_concepts,
                    chunk=chunk,
                    attribution=attribution,
                )
                all_associations.extend(associations)
            except Exception:
                logger.warning(
                    "Association discovery failed for chunk %d, skipping",
                    chunk.chunk_index,
                    exc_info=True,
                )

        return all_associations

    async def _resolve_all_conflicts(
        self,
        game_id: UUID,
        integrated_concepts: list[KGConcept],
    ) -> None:
        """Resolve conflicts for all integrated concepts.

        Checks each concept's sources and resolves priority conflicts.

        Args:
            game_id: UUID of the owning game.
            integrated_concepts: Concepts integrated into the graph.
        """
        for concept in integrated_concepts:
            try:
                sources = await self._conflicts._source_repo.list_by_concept(concept.id)
                if len(sources) > 1:
                    newest_source = max(sources, key=lambda s: s.created_at)
                    await self._conflicts.resolve_conflicts(
                        game_id=game_id,
                        concept_id=concept.id,
                        new_source=newest_source,
                    )
            except Exception:
                logger.warning(
                    "Conflict resolution failed for concept %s, skipping",
                    concept.id,
                    exc_info=True,
                )

    async def _embed_concepts(
        self,
        concepts: list[KGConcept],
    ) -> None:
        """Generate embeddings for integrated concepts.

        Uses batch embedding for efficiency. Errors are logged but
        do not fail the pipeline -- embeddings can be regenerated.

        Args:
            concepts: Concepts to embed.
        """
        if not concepts:
            return

        try:
            await self._embeddings.embed_batch(concepts)
        except Exception:
            logger.warning(
                "Embedding generation failed for %d concepts, skipping",
                len(concepts),
                exc_info=True,
            )

    async def _get_existing_concepts(
        self,
        game_id: UUID,
    ) -> list[ConceptSummary]:
        """Get existing concept summaries for a game.

        Used to provide context for cross-document association discovery.

        Args:
            game_id: UUID of the game.

        Returns:
            List of concept summaries. Empty if provider not configured.
        """
        if self._existing_concepts_provider is None:
            return []

        try:
            return await self._existing_concepts_provider(game_id)
        except Exception:
            logger.warning(
                "Failed to fetch existing concepts for game %s",
                game_id,
                exc_info=True,
            )
            return []

    @staticmethod
    def _count_cross_document(
        associations: list[DiscoveredAssociation],
        new_concepts: list[ExtractedConcept],
    ) -> int:
        """Count associations referencing concepts outside this extraction.

        A cross-document association has at least one endpoint that was
        not extracted from this document.

        Args:
            associations: All discovered associations.
            new_concepts: Newly extracted concepts.

        Returns:
            Number of cross-document associations.
        """
        new_names = {c.name.lower() for c in new_concepts}
        return sum(
            1
            for a in associations
            if (
                a.source_concept_name.lower() not in new_names
                or a.target_concept_name.lower() not in new_names
            )
        )

    def _emit_event(
        self,
        document_id: UUID,
        event_type: str,
        payload: dict[str, object],
    ) -> None:
        """Emit an SSE event if a publisher is configured.

        Args:
            document_id: UUID of the document being processed.
            event_type: Event type string.
            payload: Event data.
        """
        if self._event_publisher is not None:
            try:
                self._event_publisher.publish(document_id, event_type, payload)
            except Exception:
                logger.debug(
                    "Failed to emit event %s for document %s",
                    event_type,
                    document_id,
                    exc_info=True,
                )
