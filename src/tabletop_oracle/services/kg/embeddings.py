"""EmbeddingService implementation using configured model slot.

Calls the configured embedding model slot (OpenAI-compatible API) to
generate vector embeddings for KG concepts and search queries. Supports
batch embedding for efficiency and tracks model_id on each embedding
for re-embedding on model change.

Design reference: docs/architecture/epic-003-knowledge-graph-engine/design.md
    Component Design section 6 (Embedding Service).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import httpx

from tabletop_oracle.services.kg.models import (
    EmbeddingError,
    EmbeddingResult,
    EmbeddingUnavailableError,
)

if TYPE_CHECKING:
    from tabletop_oracle.models.knowledge_graph import KGConcept
    from tabletop_oracle.services.vision.protocols import ModelSlotRepository

logger = logging.getLogger(__name__)

_CAPABILITY = "embedding"

_MAX_RETRIES = 2
_REQUEST_TIMEOUT_SECONDS = 60.0
_MAX_BATCH_SIZE = 100


def _build_concept_text(concept: KGConcept) -> str:
    """Build embedding input text from concept attributes.

    Combines the concept name, semantic type, and description into a
    single string. This is the canonical embedding input format per the
    design doc.

    Args:
        concept: The KG concept to build text for.

    Returns:
        Combined text for embedding.
    """
    return f"{concept.name} [{concept.semantic_type}]: {concept.description}"


class EmbeddingService:
    """Generates vector embeddings for KG concepts.

    Uses the embedding model slot from the model_slots configuration
    table (PRD-007). Embeddings are stored in the concept_embeddings
    table with pgvector.

    The service calls an OpenAI-compatible embeddings API. Batch requests
    are chunked to respect API limits.

    Args:
        model_slot_repo: Repository for model slot configuration.
        http_client: Optional pre-configured httpx.AsyncClient for API calls.
        embedding_dim: Expected embedding dimensionality. Used for validation.
    """

    def __init__(
        self,
        model_slot_repo: ModelSlotRepository,
        http_client: httpx.AsyncClient | None = None,
        embedding_dim: int = 1536,
    ) -> None:
        self._model_slot_repo = model_slot_repo
        self._http_client = http_client
        self._owns_client = http_client is None
        self._embedding_dim = embedding_dim

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
        text = _build_concept_text(concept)
        result = await self._call_embedding_api([text])
        return result.embeddings[0]

    async def embed_batch(self, concepts: list[KGConcept]) -> list[list[float]]:
        """Generate embeddings for a batch of concepts.

        Batches API calls to the embedding model for efficiency. Large
        batches are split into chunks of _MAX_BATCH_SIZE to respect
        API limits. Order of returned embeddings matches input order.

        Args:
            concepts: The KG concepts to embed.

        Returns:
            List of vector embeddings, one per concept.

        Raises:
            EmbeddingError: If the embedding API call fails after retries.
            EmbeddingUnavailableError: If no embedding model is configured.
        """
        if not concepts:
            return []

        texts = [_build_concept_text(c) for c in concepts]
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), _MAX_BATCH_SIZE):
            chunk = texts[i : i + _MAX_BATCH_SIZE]
            result = await self._call_embedding_api(chunk)
            all_embeddings.extend(result.embeddings)

        return all_embeddings

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
            ValueError: If query_text is empty.
        """
        if not query_text or not query_text.strip():
            raise ValueError("query_text must not be empty")

        result = await self._call_embedding_api([query_text])
        return result.embeddings[0]

    async def get_model_id(self) -> str:
        """Get the currently configured embedding model identifier.

        Returns:
            The model_id string from the embedding model slot.

        Raises:
            EmbeddingUnavailableError: If no embedding model is configured.
        """
        slot = await self._get_model_slot()
        return str(slot["model_id"])

    async def _call_embedding_api(self, texts: list[str]) -> EmbeddingResult:
        """Call the embedding API with retry logic.

        Args:
            texts: List of texts to embed.

        Returns:
            EmbeddingResult with embeddings and metadata.

        Raises:
            EmbeddingError: If the API call fails after retries.
            EmbeddingUnavailableError: If no embedding model is configured.
        """
        slot = await self._get_model_slot()
        provider = str(slot["provider"])
        model_id = str(slot["model_id"])

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return await self._do_request(
                    provider=provider,
                    model_id=model_id,
                    texts=texts,
                )
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response.status_code < 500 and exc.response.status_code != 429:
                    break  # Non-retryable client error
                logger.warning(
                    "Embedding API call failed (attempt %d/%d)",
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    extra={
                        "status": exc.response.status_code,
                        "provider": provider,
                    },
                )
            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning(
                    "Embedding API call failed (attempt %d/%d)",
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    extra={"error": str(exc), "provider": provider},
                )

        raise EmbeddingError(
            f"Embedding API call failed after {_MAX_RETRIES + 1} attempts: {last_error}",
            provider=provider,
            model_id=model_id,
        )

    async def _do_request(
        self,
        *,
        provider: str,
        model_id: str,
        texts: list[str],
    ) -> EmbeddingResult:
        """Execute a single embedding API request.

        Calls the OpenAI-compatible /v1/embeddings endpoint. The same
        endpoint format works for OpenAI, Azure OpenAI, and compatible
        providers.

        Args:
            provider: Model provider identifier.
            model_id: Provider-specific model identifier.
            texts: List of texts to embed in this request.

        Returns:
            EmbeddingResult with parsed embeddings and metadata.

        Raises:
            httpx.HTTPError: On transport or HTTP errors.
            EmbeddingError: If the response is malformed.
        """
        client = self._get_http_client()
        payload: dict[str, Any] = {
            "model": model_id,
            "input": texts,
        }

        response = await client.post(
            "/v1/embeddings",
            json=payload,
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()

        return self._parse_response(data, model_id=model_id, provider=provider)

    def _parse_response(
        self,
        data: dict[str, Any],
        *,
        model_id: str,
        provider: str,
    ) -> EmbeddingResult:
        """Parse an OpenAI-compatible embeddings API response.

        Args:
            data: Raw JSON response from the API.
            model_id: Model identifier for attribution.
            provider: Provider identifier for error context.

        Returns:
            EmbeddingResult with extracted embeddings and token count.

        Raises:
            EmbeddingError: If the response is malformed or dimensions
                do not match the configured embedding_dim.
        """
        try:
            embedding_data = data["data"]
            # Sort by index to guarantee order matches input order
            sorted_data = sorted(embedding_data, key=lambda x: x["index"])
            embeddings = [item["embedding"] for item in sorted_data]
            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
        except (KeyError, IndexError, TypeError) as exc:
            raise EmbeddingError(
                f"Malformed embedding response: {exc}",
                provider=provider,
                model_id=model_id,
            ) from exc

        # Validate dimensionality
        for i, emb in enumerate(embeddings):
            if len(emb) != self._embedding_dim:
                raise EmbeddingError(
                    f"Embedding dimension mismatch at index {i}: "
                    f"expected {self._embedding_dim}, got {len(emb)}",
                    provider=provider,
                    model_id=model_id,
                )

        return EmbeddingResult(
            embeddings=embeddings,
            model_id=model_id,
            input_tokens=input_tokens,
        )

    async def _get_model_slot(self) -> dict[str, Any]:
        """Fetch the embedding model slot or raise.

        Returns:
            Model slot configuration dict.

        Raises:
            EmbeddingUnavailableError: If no embedding model slot is configured.
        """
        slot = await self._model_slot_repo.get_by_capability(_CAPABILITY)
        if slot is None:
            raise EmbeddingUnavailableError()
        return slot

    def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client for API calls.

        Returns:
            An httpx.AsyncClient instance.
        """
        if self._http_client is None:
            self._http_client = httpx.AsyncClient()
            self._owns_client = True
        return self._http_client
