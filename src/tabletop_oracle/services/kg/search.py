"""pgvector cosine similarity search for KG concepts.

Implements semantic search over concept embeddings using pgvector's
HNSW cosine similarity index. Used by the KGRetrievalService to find
concepts relevant to a natural-language query.

Design reference: docs/architecture/epic-003-knowledge-graph-engine/design.md
    Component Design section 7 (KG Retrieval Service), Query Patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import select, text

from tabletop_oracle.models.knowledge_graph import (
    ConceptEmbedding,
    KGConcept,
    KGConceptSource,
)

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from tabletop_oracle.services.kg.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimilarityResult:
    """A concept returned from pgvector cosine similarity search.

    Attributes:
        concept_id: UUID of the matched concept.
        name: Concept name.
        semantic_type: Concept semantic type.
        description: Concept description.
        similarity_score: Cosine similarity score (0 to 1, higher is more similar).
        model_id: The embedding model that produced the stored embedding.
    """

    concept_id: UUID
    name: str
    semantic_type: str
    description: str
    similarity_score: float
    model_id: str


async def search_similar_concepts(
    db: AsyncSession,
    embedding_service: EmbeddingService,
    game_id: UUID,
    query_text: str,
    limit: int = 20,
    active_expansion_ids: list[UUID] | None = None,
) -> list[SimilarityResult]:
    """Find concepts most similar to query text using pgvector cosine search.

    Generates an embedding for the query text, then performs cosine
    similarity search over concept embeddings scoped to the given game.
    Only canonical concepts are searched (canonical_id IS NULL).

    Session-level expansion filtering (AC-511): When active_expansion_ids
    is provided, concepts are filtered to include only those whose
    sources come from base-game documents (expansion_id IS NULL) or
    documents belonging to active expansions.

    Args:
        db: Async database session.
        embedding_service: Service for generating the query embedding.
        game_id: Game scope for the search.
        query_text: Natural-language search input.
        limit: Maximum number of results to return.
        active_expansion_ids: Active expansion IDs for session filtering.
            None means no filtering (all expansions included).

    Returns:
        Ranked list of SimilarityResult, ordered by similarity (descending).
    """
    query_embedding = await embedding_service.embed_query(query_text)

    # pgvector cosine distance operator: <=> returns distance (0 = identical).
    # We convert to similarity: 1 - distance.
    cosine_distance = ConceptEmbedding.embedding.cosine_distance(query_embedding)

    stmt = (
        select(
            KGConcept.id,
            KGConcept.name,
            KGConcept.semantic_type,
            KGConcept.description,
            (text("1") - cosine_distance).label("similarity_score"),
            ConceptEmbedding.model_id,
        )
        .join(ConceptEmbedding, ConceptEmbedding.concept_id == KGConcept.id)
        .where(
            KGConcept.game_id == game_id,
            KGConcept.canonical_id.is_(None),  # Only canonical concepts
        )
    )

    if active_expansion_ids is not None:
        expansion_filter = (
            select(KGConceptSource.concept_id)
            .where(
                KGConceptSource.concept_id == KGConcept.id,
                KGConceptSource.expansion_id.is_(None)
                | KGConceptSource.expansion_id.in_(active_expansion_ids),
            )
            .correlate(KGConcept)
            .exists()
        )
        stmt = stmt.where(expansion_filter)

    stmt = stmt.order_by(cosine_distance).limit(limit)

    result = await db.execute(stmt)
    rows = result.all()

    return [
        SimilarityResult(
            concept_id=row.id,
            name=row.name,
            semantic_type=row.semantic_type,
            description=row.description,
            similarity_score=float(row.similarity_score),
            model_id=row.model_id,
        )
        for row in rows
    ]


async def store_concept_embedding(
    db: AsyncSession,
    concept_id: UUID,
    embedding: list[float],
    model_id: str,
) -> ConceptEmbedding:
    """Store or replace the embedding for a concept.

    If an embedding already exists for the concept (from a previous
    model), it is replaced. The unique constraint on concept_id
    enforces the 1:1 relationship.

    Args:
        db: Async database session.
        concept_id: UUID of the concept.
        embedding: Vector embedding as a list of floats.
        model_id: Identifier of the model that produced this embedding.

    Returns:
        The persisted ConceptEmbedding record.
    """
    stmt = select(ConceptEmbedding).where(ConceptEmbedding.concept_id == concept_id)
    result = await db.execute(stmt)
    existing = result.scalar_one_or_none()

    if existing is not None:
        existing.embedding = embedding
        existing.model_id = model_id
        await db.flush()
        return existing

    new_embedding = ConceptEmbedding(
        concept_id=concept_id,
        embedding=embedding,
        model_id=model_id,
    )
    db.add(new_embedding)
    await db.flush()
    return new_embedding


async def store_batch_embeddings(
    db: AsyncSession,
    concept_ids: list[UUID],
    embeddings: list[list[float]],
    model_id: str,
) -> list[ConceptEmbedding]:
    """Store embeddings for a batch of concepts.

    Efficiently stores or replaces embeddings for multiple concepts.
    Uses individual upsert operations per concept (pgvector does not
    support bulk upsert natively with vectors).

    Args:
        db: Async database session.
        concept_ids: UUIDs of concepts, in order matching embeddings.
        embeddings: Vector embeddings, one per concept.
        model_id: Identifier of the model that produced these embeddings.

    Returns:
        List of persisted ConceptEmbedding records.

    Raises:
        ValueError: If concept_ids and embeddings have different lengths.
    """
    if len(concept_ids) != len(embeddings):
        raise ValueError(
            f"concept_ids length ({len(concept_ids)}) must match "
            f"embeddings length ({len(embeddings)})"
        )

    results: list[ConceptEmbedding] = []
    for concept_id, embedding in zip(concept_ids, embeddings, strict=True):
        record = await store_concept_embedding(db, concept_id, embedding, model_id)
        results.append(record)

    return results
