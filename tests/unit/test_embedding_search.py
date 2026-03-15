"""Unit tests for the pgvector search module.

Tests the store_concept_embedding, store_batch_embeddings functions with
mocked database sessions. The search_similar_concepts function requires
pgvector and is covered by integration tests.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from tabletop_oracle.services.kg.search import (
    SimilarityResult,
    store_batch_embeddings,
    store_concept_embedding,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_db_session(existing_record: MagicMock | None = None) -> AsyncMock:
    """Build a mock AsyncSession."""
    db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = existing_record
    db.execute = AsyncMock(return_value=mock_result)
    db.add = MagicMock()
    db.flush = AsyncMock()
    return db


# ---------------------------------------------------------------------------
# SimilarityResult
# ---------------------------------------------------------------------------


class TestSimilarityResult:
    """Tests for the SimilarityResult dataclass."""

    def test_frozen_cannot_mutate(self) -> None:
        result = SimilarityResult(
            concept_id=uuid.uuid4(),
            name="Test",
            semantic_type="rule",
            description="A test concept",
            similarity_score=0.95,
            model_id="text-embedding-3-small",
        )
        with pytest.raises(AttributeError):
            result.name = "Changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# store_concept_embedding
# ---------------------------------------------------------------------------


class TestStoreConceptEmbedding:
    """Tests for store_concept_embedding."""

    @pytest.mark.asyncio
    async def test_new_embedding_creates_record(self) -> None:
        db = _mock_db_session(existing_record=None)
        concept_id = uuid.uuid4()
        embedding = [0.1, 0.2, 0.3, 0.4]

        await store_concept_embedding(db, concept_id, embedding, "model-v1")

        db.add.assert_called_once()
        db.flush.assert_awaited()

    @pytest.mark.asyncio
    async def test_existing_embedding_updates_record(self) -> None:
        existing = MagicMock()
        existing.embedding = [0.0, 0.0, 0.0, 0.0]
        existing.model_id = "old-model"
        db = _mock_db_session(existing_record=existing)

        concept_id = uuid.uuid4()
        new_embedding = [0.1, 0.2, 0.3, 0.4]

        result = await store_concept_embedding(db, concept_id, new_embedding, "new-model")

        assert result is existing
        assert existing.model_id == "new-model"
        db.add.assert_not_called()


# ---------------------------------------------------------------------------
# store_batch_embeddings
# ---------------------------------------------------------------------------


class TestStoreBatchEmbeddings:
    """Tests for store_batch_embeddings."""

    @pytest.mark.asyncio
    async def test_mismatched_lengths_raises_value_error(self) -> None:
        db = _mock_db_session()

        with pytest.raises(ValueError, match="must match"):
            await store_batch_embeddings(
                db,
                concept_ids=[uuid.uuid4()],
                embeddings=[[0.1], [0.2]],
                model_id="model-v1",
            )

    @pytest.mark.asyncio
    async def test_empty_batch_returns_empty(self) -> None:
        db = _mock_db_session()

        result = await store_batch_embeddings(db, [], [], "model-v1")
        assert result == []

    @pytest.mark.asyncio
    async def test_multiple_concepts_stores_all(self) -> None:
        db = _mock_db_session(existing_record=None)
        ids = [uuid.uuid4(), uuid.uuid4()]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]

        result = await store_batch_embeddings(db, ids, embeddings, "model-v1")

        assert len(result) == 2
        assert db.add.call_count == 2
