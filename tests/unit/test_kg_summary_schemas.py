"""Unit tests for KG summary Pydantic schemas.

Tests serialisation shape, default values, and validation constraints.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from tabletop_oracle.schemas.knowledge_graph import (
    DocumentContributionResponse,
    DocumentSummary,
    KGSummaryResponse,
    ReprocessResponse,
    SemanticTypeCount,
)


class TestSemanticTypeCount:
    """Tests for SemanticTypeCount schema."""

    def test_valid_construction(self) -> None:
        """SemanticTypeCount can be constructed with valid values."""
        stc = SemanticTypeCount(type="game_mechanic", count=42)
        assert stc.type == "game_mechanic"
        assert stc.count == 42

    def test_rejects_negative_count(self) -> None:
        """SemanticTypeCount rejects negative count."""
        with pytest.raises(ValueError):
            SemanticTypeCount(type="test", count=-1)


class TestDocumentSummary:
    """Tests for DocumentSummary schema."""

    def test_valid_construction(self) -> None:
        """DocumentSummary can be constructed with valid values."""
        doc_id = uuid4()
        ds = DocumentSummary(
            document_id=doc_id,
            name="Rules",
            type="core_rules",
            concepts_contributed=10,
            associations_contributed=20,
        )
        assert ds.document_id == doc_id
        assert ds.name == "Rules"
        assert ds.concepts_contributed == 10


class TestKGSummaryResponse:
    """Tests for KGSummaryResponse schema."""

    def test_serialises_to_expected_shape(self) -> None:
        """KGSummaryResponse serialises to the design contract shape."""
        game_id = uuid4()
        doc_id = uuid4()
        now = datetime(2026, 3, 14, 12, 30, 0, tzinfo=UTC)

        response = KGSummaryResponse(
            game_id=game_id,
            concept_count=50,
            association_count=100,
            document_count=2,
            semantic_types=[SemanticTypeCount(type="rule", count=50)],
            documents=[
                DocumentSummary(
                    document_id=doc_id,
                    name="Rules",
                    type="core_rules",
                    concepts_contributed=50,
                    associations_contributed=100,
                )
            ],
            last_updated_at=now,
        )

        data = response.model_dump(mode="json")
        assert data["game_id"] == str(game_id)
        assert data["concept_count"] == 50
        assert data["semantic_types"][0]["type"] == "rule"
        assert data["documents"][0]["document_id"] == str(doc_id)

    def test_last_updated_at_nullable(self) -> None:
        """KGSummaryResponse accepts None for last_updated_at."""
        response = KGSummaryResponse(
            game_id=uuid4(),
            concept_count=0,
            association_count=0,
            document_count=0,
            semantic_types=[],
            documents=[],
            last_updated_at=None,
        )
        assert response.last_updated_at is None


class TestDocumentContributionResponse:
    """Tests for DocumentContributionResponse schema."""

    def test_serialises_with_expansion(self) -> None:
        """DocumentContributionResponse serialises expansion_id correctly."""
        doc_id = uuid4()
        exp_id = uuid4()
        now = datetime(2026, 3, 14, 12, 30, 0, tzinfo=UTC)

        response = DocumentContributionResponse(
            document_id=doc_id,
            name="Expansion FAQ",
            type="faq",
            expansion_id=exp_id,
            concepts_contributed=5,
            associations_contributed=10,
            conflicts_detected=1,
            last_processed_at=now,
        )

        data = response.model_dump(mode="json")
        assert data["expansion_id"] == str(exp_id)
        assert data["conflicts_detected"] == 1

    def test_expansion_id_nullable(self) -> None:
        """DocumentContributionResponse accepts None for expansion_id."""
        response = DocumentContributionResponse(
            document_id=uuid4(),
            name="Rules",
            type="core_rules",
            expansion_id=None,
            concepts_contributed=10,
            associations_contributed=20,
            conflicts_detected=0,
            last_processed_at=None,
        )
        assert response.expansion_id is None
        assert response.last_processed_at is None


class TestReprocessResponse:
    """Tests for ReprocessResponse schema."""

    def test_default_status(self) -> None:
        """ReprocessResponse defaults status to 'reprocessing'."""
        response = ReprocessResponse(
            game_id=uuid4(),
            documents_queued=5,
        )
        assert response.status == "reprocessing"

    def test_serialises_correctly(self) -> None:
        """ReprocessResponse serialises to expected shape."""
        game_id = uuid4()
        response = ReprocessResponse(
            game_id=game_id,
            documents_queued=3,
        )
        data = response.model_dump(mode="json")
        assert data["game_id"] == str(game_id)
        assert data["documents_queued"] == 3
        assert data["status"] == "reprocessing"
