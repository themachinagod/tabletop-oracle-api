"""Unit tests for the EmbeddingService.

Tests all service methods with mocked dependencies (ModelSlotRepository,
HTTP client). Covers happy paths, batch embedding, query embedding,
error handling, retry logic, input validation, and dimension validation.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tabletop_oracle.services.kg.embeddings import (
    EmbeddingService,
    _build_concept_text,
)
from tabletop_oracle.services.kg.models import (
    EmbeddingError,
    EmbeddingUnavailableError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 4  # Use small dimension for tests


def _make_slot(**overrides: Any) -> dict[str, Any]:
    """Build a model slot configuration dict with sensible defaults."""
    defaults: dict[str, Any] = {
        "provider": "openai",
        "model_id": "text-embedding-3-small",
    }
    defaults.update(overrides)
    return defaults


def _make_embedding_response(
    embeddings: list[list[float]] | None = None,
    prompt_tokens: int = 10,
) -> dict[str, Any]:
    """Build a mock OpenAI-compatible embedding API response."""
    if embeddings is None:
        embeddings = [[0.1, 0.2, 0.3, 0.4]]
    return {
        "data": [
            {"index": i, "embedding": emb}
            for i, emb in enumerate(embeddings)
        ],
        "usage": {"prompt_tokens": prompt_tokens},
    }


def _make_concept(
    name: str = "Plasma Cannon",
    semantic_type: str = "weapon",
    description: str = "A high-energy ranged weapon",
) -> MagicMock:
    """Build a mock KGConcept."""
    concept = MagicMock()
    concept.name = name
    concept.semantic_type = semantic_type
    concept.description = description
    return concept


def _mock_model_slot_repo(slot: dict[str, Any] | None = None) -> AsyncMock:
    """Build a mock ModelSlotRepository."""
    repo = AsyncMock()
    repo.get_by_capability = AsyncMock(return_value=slot)
    return repo


def _mock_http_client(
    response_json: dict[str, Any],
    status_code: int = 200,
) -> AsyncMock:
    """Build a mock httpx.AsyncClient that returns a fixed response."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = status_code
    mock_response.json.return_value = response_json
    mock_response.raise_for_status = MagicMock()
    if status_code >= 400:
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error",
            request=MagicMock(spec=httpx.Request),
            response=mock_response,
        )

    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(return_value=mock_response)
    return client


# ---------------------------------------------------------------------------
# _build_concept_text
# ---------------------------------------------------------------------------


class TestBuildConceptText:
    """Tests for the _build_concept_text helper function."""

    def test_combines_name_type_description(self) -> None:
        concept = _make_concept(
            name="Plasma Cannon",
            semantic_type="weapon",
            description="A devastating ranged weapon",
        )
        result = _build_concept_text(concept)
        assert result == "Plasma Cannon [weapon]: A devastating ranged weapon"

    def test_handles_special_characters(self) -> None:
        concept = _make_concept(
            name="Hit & Run",
            semantic_type="game_mechanic",
            description='Move after shooting (aka "shoot & scoot")',
        )
        result = _build_concept_text(concept)
        assert "Hit & Run" in result
        assert "[game_mechanic]" in result


# ---------------------------------------------------------------------------
# EmbeddingService.embed_concept
# ---------------------------------------------------------------------------


class TestEmbedConcept:
    """Tests for EmbeddingService.embed_concept."""

    @pytest.mark.asyncio
    async def test_embed_concept_returns_embedding_vector(self) -> None:
        embedding = [0.1, 0.2, 0.3, 0.4]
        repo = _mock_model_slot_repo(_make_slot())
        client = _mock_http_client(_make_embedding_response([embedding]))

        service = EmbeddingService(repo, http_client=client, embedding_dim=_DIM)
        result = await service.embed_concept(_make_concept())

        assert result == embedding

    @pytest.mark.asyncio
    async def test_embed_concept_sends_correct_payload(self) -> None:
        repo = _mock_model_slot_repo(_make_slot())
        client = _mock_http_client(_make_embedding_response())

        service = EmbeddingService(repo, http_client=client, embedding_dim=_DIM)
        await service.embed_concept(_make_concept())

        call_kwargs = client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["model"] == "text-embedding-3-small"
        assert len(payload["input"]) == 1
        assert "Plasma Cannon [weapon]:" in payload["input"][0]

    @pytest.mark.asyncio
    async def test_embed_concept_no_slot_raises_unavailable(self) -> None:
        repo = _mock_model_slot_repo(None)
        service = EmbeddingService(repo, embedding_dim=_DIM)

        with pytest.raises(EmbeddingUnavailableError):
            await service.embed_concept(_make_concept())


# ---------------------------------------------------------------------------
# EmbeddingService.embed_batch
# ---------------------------------------------------------------------------


class TestEmbedBatch:
    """Tests for EmbeddingService.embed_batch."""

    @pytest.mark.asyncio
    async def test_embed_batch_empty_list_returns_empty(self) -> None:
        repo = _mock_model_slot_repo(_make_slot())
        service = EmbeddingService(repo, embedding_dim=_DIM)

        result = await service.embed_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_batch_multiple_concepts_returns_all(self) -> None:
        emb1 = [0.1, 0.2, 0.3, 0.4]
        emb2 = [0.5, 0.6, 0.7, 0.8]
        repo = _mock_model_slot_repo(_make_slot())
        client = _mock_http_client(_make_embedding_response([emb1, emb2]))

        service = EmbeddingService(repo, http_client=client, embedding_dim=_DIM)
        concepts = [_make_concept("A"), _make_concept("B")]
        result = await service.embed_batch(concepts)

        assert len(result) == 2
        assert result[0] == emb1
        assert result[1] == emb2

    @pytest.mark.asyncio
    async def test_embed_batch_large_batch_chunks_requests(self) -> None:
        """Verify that batches larger than _MAX_BATCH_SIZE are split."""
        embedding = [0.1, 0.2, 0.3, 0.4]
        repo = _mock_model_slot_repo(_make_slot())

        response = _make_embedding_response([embedding])
        client = _mock_http_client(response)

        with patch(
            "tabletop_oracle.services.kg.embeddings._MAX_BATCH_SIZE", 2
        ):
            service = EmbeddingService(repo, http_client=client, embedding_dim=_DIM)
            concepts = [_make_concept(f"C{i}") for i in range(5)]
            result = await service.embed_batch(concepts)

        # 5 concepts with batch size 2 = 3 API calls
        assert client.post.call_count == 3
        # Each call returns 1 embedding from our mock, so we get 3 total
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_embed_batch_response_order_preserved(self) -> None:
        """Verify embeddings are sorted by index even if API returns out of order."""
        repo = _mock_model_slot_repo(_make_slot())

        response_json = {
            "data": [
                {"index": 1, "embedding": [0.5, 0.6, 0.7, 0.8]},
                {"index": 0, "embedding": [0.1, 0.2, 0.3, 0.4]},
            ],
            "usage": {"prompt_tokens": 10},
        }
        client = _mock_http_client(response_json)

        service = EmbeddingService(repo, http_client=client, embedding_dim=_DIM)
        concepts = [_make_concept("A"), _make_concept("B")]
        result = await service.embed_batch(concepts)

        assert result[0] == [0.1, 0.2, 0.3, 0.4]
        assert result[1] == [0.5, 0.6, 0.7, 0.8]


# ---------------------------------------------------------------------------
# EmbeddingService.embed_query
# ---------------------------------------------------------------------------


class TestEmbedQuery:
    """Tests for EmbeddingService.embed_query."""

    @pytest.mark.asyncio
    async def test_embed_query_returns_embedding(self) -> None:
        embedding = [0.1, 0.2, 0.3, 0.4]
        repo = _mock_model_slot_repo(_make_slot())
        client = _mock_http_client(_make_embedding_response([embedding]))

        service = EmbeddingService(repo, http_client=client, embedding_dim=_DIM)
        result = await service.embed_query("How does combat work?")

        assert result == embedding

    @pytest.mark.asyncio
    async def test_embed_query_empty_raises_value_error(self) -> None:
        repo = _mock_model_slot_repo(_make_slot())
        service = EmbeddingService(repo, embedding_dim=_DIM)

        with pytest.raises(ValueError, match="must not be empty"):
            await service.embed_query("")

    @pytest.mark.asyncio
    async def test_embed_query_whitespace_only_raises_value_error(self) -> None:
        repo = _mock_model_slot_repo(_make_slot())
        service = EmbeddingService(repo, embedding_dim=_DIM)

        with pytest.raises(ValueError, match="must not be empty"):
            await service.embed_query("   ")


# ---------------------------------------------------------------------------
# EmbeddingService.get_model_id
# ---------------------------------------------------------------------------


class TestGetModelId:
    """Tests for EmbeddingService.get_model_id."""

    @pytest.mark.asyncio
    async def test_get_model_id_returns_configured_id(self) -> None:
        repo = _mock_model_slot_repo(_make_slot(model_id="text-embedding-3-large"))
        service = EmbeddingService(repo, embedding_dim=_DIM)

        result = await service.get_model_id()
        assert result == "text-embedding-3-large"

    @pytest.mark.asyncio
    async def test_get_model_id_no_slot_raises_unavailable(self) -> None:
        repo = _mock_model_slot_repo(None)
        service = EmbeddingService(repo, embedding_dim=_DIM)

        with pytest.raises(EmbeddingUnavailableError):
            await service.get_model_id()


# ---------------------------------------------------------------------------
# Error handling and retries
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling and retry behavior."""

    @pytest.mark.asyncio
    async def test_server_error_retries_and_fails(self) -> None:
        repo = _mock_model_slot_repo(_make_slot())
        client = _mock_http_client({}, status_code=500)

        service = EmbeddingService(repo, http_client=client, embedding_dim=_DIM)

        with pytest.raises(EmbeddingError):
            await service.embed_concept(_make_concept())

        # 3 attempts total (1 + 2 retries)
        assert client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_client_error_no_retry(self) -> None:
        repo = _mock_model_slot_repo(_make_slot())
        client = _mock_http_client({}, status_code=400)

        service = EmbeddingService(repo, http_client=client, embedding_dim=_DIM)

        with pytest.raises(EmbeddingError):
            await service.embed_concept(_make_concept())

        # Only 1 attempt -- client errors are not retried
        assert client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_rate_limit_429_retries(self) -> None:
        repo = _mock_model_slot_repo(_make_slot())
        client = _mock_http_client({}, status_code=429)

        service = EmbeddingService(repo, http_client=client, embedding_dim=_DIM)

        with pytest.raises(EmbeddingError):
            await service.embed_concept(_make_concept())

        # 429 is retried
        assert client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_malformed_response_raises_embedding_error(self) -> None:
        repo = _mock_model_slot_repo(_make_slot())
        client = _mock_http_client({"unexpected": "format"})

        service = EmbeddingService(repo, http_client=client, embedding_dim=_DIM)

        with pytest.raises(EmbeddingError, match="Malformed"):
            await service.embed_concept(_make_concept())


# ---------------------------------------------------------------------------
# Dimension validation
# ---------------------------------------------------------------------------


class TestDimensionValidation:
    """Tests for embedding dimension validation."""

    @pytest.mark.asyncio
    async def test_wrong_dimension_raises_embedding_error(self) -> None:
        repo = _mock_model_slot_repo(_make_slot())
        # Return 3-dim embedding when service expects 4
        client = _mock_http_client(_make_embedding_response([[0.1, 0.2, 0.3]]))

        service = EmbeddingService(repo, http_client=client, embedding_dim=_DIM)

        with pytest.raises(EmbeddingError, match="dimension mismatch"):
            await service.embed_concept(_make_concept())

    @pytest.mark.asyncio
    async def test_correct_dimension_passes(self) -> None:
        repo = _mock_model_slot_repo(_make_slot())
        client = _mock_http_client(
            _make_embedding_response([[0.1, 0.2, 0.3, 0.4]])
        )

        service = EmbeddingService(repo, http_client=client, embedding_dim=_DIM)
        result = await service.embed_concept(_make_concept())

        assert len(result) == _DIM


# ---------------------------------------------------------------------------
# Exception models
# ---------------------------------------------------------------------------


class TestExceptions:
    """Tests for exception attributes."""

    def test_embedding_error_carries_provider_and_model(self) -> None:
        err = EmbeddingError(
            "test error", provider="openai", model_id="text-embedding-3-small"
        )
        assert err.provider == "openai"
        assert err.model_id == "text-embedding-3-small"
        assert "test error" in str(err)

    def test_embedding_unavailable_error_default_message(self) -> None:
        err = EmbeddingUnavailableError()
        assert "not configured" in str(err)

    def test_embedding_unavailable_error_custom_message(self) -> None:
        err = EmbeddingUnavailableError("custom msg")
        assert "custom msg" in str(err)
