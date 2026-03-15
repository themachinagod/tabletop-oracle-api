"""Graph integration service for merging extracted concepts into the KG.

Integrates new concepts and associations into a game's knowledge graph,
handling deduplication, alias resolution, source reference tracking,
and document contribution removal.

Design reference: docs/architecture/epic-003-knowledge-graph-engine/design.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tabletop_oracle.models.knowledge_graph import (
    KGAssociation,
    KGAssociationSource,
    KGConcept,
    KGConceptSource,
)

if TYPE_CHECKING:
    from uuid import UUID

    from tabletop_oracle.repositories.kg_repository import (
        KGAssociationRepository,
        KGConceptRepository,
        KGConceptSourceRepository,
    )
    from tabletop_oracle.services.kg.associations import DiscoveredAssociation
    from tabletop_oracle.services.kg.extraction import ExtractedConcept

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RemovalResult:
    """Result of removing a document's contributions from the KG.

    Attributes:
        concepts_removed: Number of concepts entirely deleted (sole-sourced).
        concepts_updated: Number of concepts that lost a source but were
            retained because other sources still reference them.
    """

    concepts_removed: int
    concepts_updated: int


class GraphIntegrationService:
    """Integrates new concepts and associations into a game's knowledge graph.

    For concepts: if a concept with the same name already exists, adds a
    new source reference. If new, creates the concept and its first source.
    Synonym aliases are resolved to canonical concepts via ``canonical_id``.

    For associations: resolves concept names to IDs, deduplicates against
    existing associations, and creates new association-source links.

    Args:
        concept_repo: Repository for KGConcept CRUD operations.
        association_repo: Repository for KGAssociation CRUD operations.
        source_repo: Repository for KGConceptSource CRUD operations.
    """

    def __init__(
        self,
        concept_repo: KGConceptRepository,
        association_repo: KGAssociationRepository,
        source_repo: KGConceptSourceRepository,
    ) -> None:
        self._concept_repo = concept_repo
        self._association_repo = association_repo
        self._source_repo = source_repo

    async def integrate_concepts(
        self,
        game_id: UUID,
        document_id: UUID,
        version_id: UUID,
        document_type: str,
        document_name: str,
        expansion_id: UUID | None,
        extracted_concepts: list[ExtractedConcept],
    ) -> list[KGConcept]:
        """Integrate extracted concepts into the knowledge graph.

        For each concept:
        1. Check if a concept with the same name already exists in this game.
        2. If exists: add a new source reference (dedup by chunk_id).
        3. If new: create the concept, then create alias concepts for
           each alias name pointing back via ``canonical_id``.

        Returns the list of canonical concepts (new or existing) that
        were integrated, plus a concept_map for downstream use.

        Args:
            game_id: UUID of the owning game.
            document_id: UUID of the source document.
            version_id: UUID of the document version.
            document_type: Document type string for source priority.
            document_name: Display name of the source document.
            expansion_id: Optional expansion UUID.
            extracted_concepts: Concepts extracted from document chunks.

        Returns:
            List of integrated KGConcept entities (canonical only).
        """
        integrated: list[KGConcept] = []

        for extracted in extracted_concepts:
            concept = await self._integrate_single_concept(
                game_id=game_id,
                document_id=document_id,
                version_id=version_id,
                document_type=document_type,
                document_name=document_name,
                expansion_id=expansion_id,
                extracted=extracted,
            )
            integrated.append(concept)

        logger.info(
            "Integrated %d concepts for game %s from document %s",
            len(integrated),
            game_id,
            document_id,
        )
        return integrated

    async def integrate_associations(
        self,
        game_id: UUID,
        document_id: UUID,
        associations: list[DiscoveredAssociation],
        concept_map: dict[str, UUID],
        expansion_id: UUID | None = None,
    ) -> list[KGAssociation]:
        """Integrate discovered associations into the knowledge graph.

        For each association:
        1. Resolve source and target concept names to UUIDs via concept_map.
        2. Check if the association already exists (same game, source,
           target, label).
        3. If exists: add a new source reference (dedup by chunk_id).
        4. If new: create the association with its first source.

        Args:
            game_id: UUID of the owning game.
            document_id: UUID of the source document.
            associations: Associations discovered from document chunks.
            concept_map: Mapping from concept name (lowercase) to UUID.
            expansion_id: Optional expansion UUID.

        Returns:
            List of integrated KGAssociation entities.
        """
        integrated: list[KGAssociation] = []

        for discovered in associations:
            association = await self._integrate_single_association(
                game_id=game_id,
                document_id=document_id,
                discovered=discovered,
                concept_map=concept_map,
                expansion_id=expansion_id,
            )
            if association is not None:
                integrated.append(association)

        logger.info(
            "Integrated %d associations for game %s",
            len(integrated),
            game_id,
        )
        return integrated

    def build_concept_map(
        self,
        concepts: list[KGConcept],
    ) -> dict[str, UUID]:
        """Build a name-to-UUID mapping from integrated concepts.

        Maps lowercase canonical names to concept IDs. Used by
        ``integrate_associations`` to resolve concept references.

        Args:
            concepts: List of integrated canonical concepts.

        Returns:
            Dict mapping lowercase concept names to UUIDs.
        """
        return {c.name.lower(): c.id for c in concepts}

    async def remove_document_contributions(
        self,
        game_id: UUID,
        document_id: UUID,
    ) -> RemovalResult:
        """Remove a document's contributions from the knowledge graph.

        For each concept sourced from this document:
        - If this is the sole source: delete the concept (cascades to
          sources, embeddings, and associations via FK ON DELETE CASCADE).
        - If other sources remain: remove only this document's source
          reference.

        After source cleanup, orphaned associations (where either endpoint
        concept was removed) are cleaned up by the cascade.

        Args:
            game_id: UUID of the owning game.
            document_id: UUID of the document to remove.

        Returns:
            RemovalResult with counts of removed and updated concepts.
        """
        concepts_removed = 0
        concepts_updated = 0

        # Find all source references from this document
        sources = await self._source_repo.list_by_document(
            document_id=document_id,
        )

        # Group sources by concept_id
        concept_sources: dict[UUID, list[KGConceptSource]] = {}
        for source in sources:
            concept_sources.setdefault(source.concept_id, []).append(source)

        for concept_id, doc_sources in concept_sources.items():
            # Get all sources for this concept (not just from this document)
            all_sources = await self._source_repo.list_by_concept(concept_id)

            non_doc_sources = [s for s in all_sources if s.document_id != document_id]

            if not non_doc_sources:
                # Sole source -- delete the concept (cascade handles rest)
                await self._concept_repo.delete(concept_id)
                concepts_removed += 1
            else:
                # Remove only this document's source references
                for source in doc_sources:
                    await self._source_repo.delete(source.id)
                concepts_updated += 1

        logger.info(
            "Removed document %s contributions from game %s: "
            "concepts_removed=%d concepts_updated=%d",
            document_id,
            game_id,
            concepts_removed,
            concepts_updated,
        )
        return RemovalResult(
            concepts_removed=concepts_removed,
            concepts_updated=concepts_updated,
        )

    async def _integrate_single_concept(
        self,
        game_id: UUID,
        document_id: UUID,
        version_id: UUID,
        document_type: str,
        document_name: str,
        expansion_id: UUID | None,
        extracted: ExtractedConcept,
    ) -> KGConcept:
        """Integrate a single extracted concept.

        Handles deduplication (existing concept gets new source),
        new concept creation, and alias creation.

        Args:
            game_id: UUID of the owning game.
            document_id: UUID of the source document.
            version_id: UUID of the document version.
            document_type: Document type string.
            document_name: Display name of the document.
            expansion_id: Optional expansion UUID.
            extracted: The extracted concept to integrate.

        Returns:
            The canonical KGConcept (existing or newly created).
        """
        existing = await self._concept_repo.find_by_name(game_id, extracted.name)

        if existing is not None:
            # Resolve to canonical concept if this is an alias
            canonical = await self._resolve_canonical(existing)
            await self._add_source_if_new(
                concept_id=canonical.id,
                document_id=document_id,
                version_id=version_id,
                chunk_id=extracted.source_chunk_id,
                document_type=document_type,
                document_name=document_name,
                expansion_id=expansion_id,
                source_text=extracted.source_text,
            )
            logger.debug(
                "Merged concept '%s' into existing '%s'",
                extracted.name,
                canonical.name,
            )
            return canonical

        # New concept
        concept = KGConcept(
            game_id=game_id,
            name=extracted.name,
            semantic_type=extracted.semantic_type,
            description=extracted.description,
        )
        concept = await self._concept_repo.create(concept)

        # Create source reference
        await self._create_source(
            concept_id=concept.id,
            document_id=document_id,
            version_id=version_id,
            chunk_id=extracted.source_chunk_id,
            document_type=document_type,
            document_name=document_name,
            expansion_id=expansion_id,
            source_text=extracted.source_text,
        )

        # Create alias concepts for synonyms
        for alias_name in extracted.aliases:
            alias_existing = await self._concept_repo.find_by_name(game_id, alias_name)
            if alias_existing is None:
                alias_concept = KGConcept(
                    game_id=game_id,
                    name=alias_name,
                    semantic_type=extracted.semantic_type,
                    description=f"Alias for {extracted.name}",
                    canonical_id=concept.id,
                )
                await self._concept_repo.create(alias_concept)
                logger.debug(
                    "Created alias '%s' -> '%s'",
                    alias_name,
                    extracted.name,
                )

        return concept

    async def _resolve_canonical(self, concept: KGConcept) -> KGConcept:
        """Follow canonical_id chain to the root canonical concept.

        Args:
            concept: The concept to resolve.

        Returns:
            The canonical concept (may be the same concept if already canonical).
        """
        if concept.canonical_id is None:
            return concept

        canonical = await self._concept_repo.get_by_id(concept.canonical_id)
        if canonical is None:
            # Broken chain — treat the concept itself as canonical
            logger.warning(
                "Broken canonical chain: concept %s references missing %s",
                concept.id,
                concept.canonical_id,
            )
            return concept
        return canonical

    async def _add_source_if_new(
        self,
        concept_id: UUID,
        document_id: UUID,
        version_id: UUID,
        chunk_id: UUID,
        document_type: str,
        document_name: str,
        expansion_id: UUID | None,
        source_text: str,
    ) -> KGConceptSource | None:
        """Add a source reference if one doesn't already exist for this chunk.

        Args:
            concept_id: UUID of the concept.
            document_id: UUID of the source document.
            version_id: UUID of the document version.
            chunk_id: UUID of the source chunk.
            document_type: Document type string.
            document_name: Display name of the document.
            expansion_id: Optional expansion UUID.
            source_text: Verbatim text excerpt.

        Returns:
            The new source if created, None if already existed.
        """
        existing = await self._source_repo.find_by_concept_and_chunk(concept_id, chunk_id)
        if existing is not None:
            return None

        return await self._create_source(
            concept_id=concept_id,
            document_id=document_id,
            version_id=version_id,
            chunk_id=chunk_id,
            document_type=document_type,
            document_name=document_name,
            expansion_id=expansion_id,
            source_text=source_text,
        )

    async def _create_source(
        self,
        concept_id: UUID,
        document_id: UUID,
        version_id: UUID,
        chunk_id: UUID,
        document_type: str,
        document_name: str,
        expansion_id: UUID | None,
        source_text: str,
    ) -> KGConceptSource:
        """Create a concept source reference.

        Args:
            concept_id: UUID of the concept.
            document_id: UUID of the source document.
            version_id: UUID of the document version.
            chunk_id: UUID of the source chunk.
            document_type: Document type string.
            document_name: Display name of the document.
            expansion_id: Optional expansion UUID.
            source_text: Verbatim text excerpt.

        Returns:
            The persisted KGConceptSource entity.
        """
        source = KGConceptSource(
            concept_id=concept_id,
            document_id=document_id,
            version_id=version_id,
            chunk_id=chunk_id,
            document_type=document_type,
            document_name=document_name,
            expansion_id=expansion_id,
            source_text=source_text,
        )
        return await self._source_repo.create(source)

    async def _integrate_single_association(
        self,
        game_id: UUID,
        document_id: UUID,
        discovered: DiscoveredAssociation,
        concept_map: dict[str, UUID],
        expansion_id: UUID | None,
    ) -> KGAssociation | None:
        """Integrate a single discovered association.

        Resolves concept names to IDs, handles deduplication.

        Args:
            game_id: UUID of the owning game.
            document_id: UUID of the source document.
            discovered: The discovered association to integrate.
            concept_map: Name-to-UUID mapping for concept resolution.
            expansion_id: Optional expansion UUID.

        Returns:
            The KGAssociation if integrated, None if concepts unresolvable.
        """
        source_id = concept_map.get(discovered.source_concept_name.lower())
        target_id = concept_map.get(discovered.target_concept_name.lower())

        if source_id is None or target_id is None:
            logger.warning(
                "Skipping association '%s' -> '%s': unresolvable concept(s)",
                discovered.source_concept_name,
                discovered.target_concept_name,
            )
            return None

        # Check for existing association
        existing = await self._association_repo.find_existing(
            game_id=game_id,
            source_concept_id=source_id,
            target_concept_id=target_id,
            relationship_label=discovered.relationship_label,
        )

        if existing is not None:
            # Add source if not already present for this chunk
            existing_source = await self._association_repo.find_source_by_chunk(
                association_id=existing.id,
                chunk_id=discovered.source_chunk_id,
            )
            if existing_source is None:
                assoc_source = KGAssociationSource(
                    association_id=existing.id,
                    document_id=document_id,
                    chunk_id=discovered.source_chunk_id,
                    expansion_id=expansion_id,
                    source_text=discovered.source_text,
                )
                await self._association_repo.add_source(assoc_source)
            return existing

        # Create new association
        association = KGAssociation(
            game_id=game_id,
            source_concept_id=source_id,
            target_concept_id=target_id,
            relationship_label=discovered.relationship_label,
            description=discovered.description,
        )
        association = await self._association_repo.create(association)

        # Add source
        assoc_source = KGAssociationSource(
            association_id=association.id,
            document_id=document_id,
            chunk_id=discovered.source_chunk_id,
            expansion_id=expansion_id,
            source_text=discovered.source_text,
        )
        await self._association_repo.add_source(assoc_source)

        return association
