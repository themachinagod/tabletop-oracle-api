"""Protocol definitions for KG service abstractions.

Defines the EmbeddingServiceProtocol that both the V1 implementation and
any future implementations must satisfy. Follows the same pattern as
the VisionService protocol layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from tabletop_oracle.models.knowledge_graph import KGConcept


@runtime_checkable
class EmbeddingServiceProtocol(Protocol):
    """Interface for concept embedding services.

    Implementations generate vector embeddings for KG concepts and
    search queries using a configured embedding model. Used by the
    KG handoff pipeline and KG retrieval service.
    """

    async def embed_concept(self, concept: KGConcept) -> list[float]:
        """Generate an embedding for a single concept.

        The embedding input combines the concept name, semantic type,
        and description into a single text for embedding.

        Args:
            concept: The KG concept to embed.

        Returns:
            Vector embedding as a list of floats.

        Raises:
            EmbeddingError: If the embedding API call fails after retries.
            EmbeddingUnavailableError: If no embedding model is configured.
        """
        ...

    async def embed_batch(self, concepts: list[KGConcept]) -> list[list[float]]:
        """Generate embeddings for a batch of concepts.

        Batches API calls to the embedding model for efficiency.
        Order of returned embeddings matches the order of input concepts.

        Args:
            concepts: The KG concepts to embed.

        Returns:
            List of vector embeddings, one per concept.

        Raises:
            EmbeddingError: If the embedding API call fails after retries.
            EmbeddingUnavailableError: If no embedding model is configured.
        """
        ...

    async def embed_query(self, query_text: str) -> list[float]:
        """Generate an embedding for a search query.

        Used by KGRetrievalService for semantic search.

        Args:
            query_text: Natural-language search input.

        Returns:
            Vector embedding as a list of floats.

        Raises:
            EmbeddingError: If the embedding API call fails after retries.
            EmbeddingUnavailableError: If no embedding model is configured.
        """
        ...
