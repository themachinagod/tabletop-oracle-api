"""Pydantic schemas for curator-facing KG summary API responses.

Follows the F001 envelope convention. Response shapes match the design
contract in docs/architecture/epic-003-knowledge-graph-engine/design.md.
"""

from __future__ import annotations

import uuid  # noqa: TC003 — Pydantic needs UUID at runtime for validation
from datetime import datetime  # noqa: TC003 — Pydantic needs datetime at runtime

from pydantic import BaseModel, Field


class SemanticTypeCount(BaseModel):
    """Count of concepts by semantic type.

    Attributes:
        type: Semantic type label (e.g., "game_mechanic", "rule").
        count: Number of concepts with this semantic type.
    """

    type: str
    count: int = Field(ge=0)


class DocumentSummary(BaseModel):
    """Per-document contribution summary within the KG summary.

    Attributes:
        document_id: UUID of the contributing document.
        name: Display name of the document.
        type: Document type classification (e.g., "core_rules").
        concepts_contributed: Number of concepts sourced from this document.
        associations_contributed: Number of associations sourced from this document.
    """

    document_id: uuid.UUID
    name: str
    type: str
    concepts_contributed: int = Field(ge=0)
    associations_contributed: int = Field(ge=0)


class KGSummaryResponse(BaseModel):
    """Aggregate KG metrics for a game (AC-510).

    Attributes:
        game_id: UUID of the game.
        concept_count: Total number of canonical concepts.
        association_count: Total number of associations.
        document_count: Number of documents contributing to the KG.
        semantic_types: Concept counts grouped by semantic type.
        documents: Per-document contribution summaries.
        last_updated_at: Most recent concept or association update timestamp.
    """

    game_id: uuid.UUID
    concept_count: int = Field(ge=0)
    association_count: int = Field(ge=0)
    document_count: int = Field(ge=0)
    semantic_types: list[SemanticTypeCount]
    documents: list[DocumentSummary]
    last_updated_at: datetime | None


class DocumentContributionResponse(BaseModel):
    """Per-document contribution detail for the documents endpoint.

    Attributes:
        document_id: UUID of the document.
        name: Display name of the document.
        type: Document type classification.
        expansion_id: Expansion UUID if the document belongs to an expansion.
        concepts_contributed: Number of concepts sourced from this document.
        associations_contributed: Number of associations sourced from this document.
        conflicts_detected: Number of conflict events for this document.
        last_processed_at: Most recent processing timestamp for this document.
    """

    document_id: uuid.UUID
    name: str
    type: str
    expansion_id: uuid.UUID | None
    concepts_contributed: int = Field(ge=0)
    associations_contributed: int = Field(ge=0)
    conflicts_detected: int = Field(ge=0)
    last_processed_at: datetime | None


class ReprocessResponse(BaseModel):
    """Response for the KG reprocess trigger endpoint.

    Attributes:
        game_id: UUID of the game being reprocessed.
        documents_queued: Number of documents queued for reprocessing.
        status: Current reprocessing status (always "reprocessing").
    """

    game_id: uuid.UUID
    documents_queued: int = Field(ge=0)
    status: str = "reprocessing"
