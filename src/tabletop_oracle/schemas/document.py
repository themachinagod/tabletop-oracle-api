"""Pydantic schemas for Document resources.

Covers request schemas (upload metadata, type update, expansion update),
response schemas (single, detail with aggregated counts), and filter/query
schemas. All field constraints and types match the design doc (EPIC-002)
and F001 conventions.
"""

from __future__ import annotations

import uuid  # noqa: TC003
from datetime import datetime  # noqa: TC003
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, model_validator


class DocumentTypeFilter(StrEnum):
    """Document type values for API schema use.

    Mirrors ``models.enums.DocumentType`` values as a StrEnum for
    Pydantic schema serialization (string values in JSON).
    """

    CORE_RULES = "core_rules"
    FAQ = "faq"
    ERRATA = "errata"
    EXPANSION_RULES = "expansion_rules"
    STRATEGY = "strategy"
    OTHER = "other"


class DocumentFormatFilter(StrEnum):
    """Supported document formats for API schema use.

    Mirrors ``models.enums.DocumentFormat``.
    """

    PDF = "pdf"
    MARKDOWN = "markdown"
    TEXT = "text"
    HTML = "html"
    DOCX = "docx"


class DocumentStatusFilter(StrEnum):
    """Document processing status for API schema use.

    Mirrors ``models.enums.DocumentStatus``.
    """

    UPLOADED = "uploaded"
    PARSING = "parsing"
    PROCESSED = "processed"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class DocumentTypeUpdate(BaseModel):
    """Request schema for reclassifying a document's type.

    Attributes:
        type: New document classification.
    """

    type: DocumentTypeFilter


class DocumentExpansionUpdate(BaseModel):
    """Request schema for changing a document's expansion association.

    Attributes:
        expansion_id: New expansion UUID, or null to associate with base game.
    """

    expansion_id: uuid.UUID | None


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class DocumentResponse(BaseModel):
    """Response schema for a document resource.

    Excludes internal ``file_path`` for security (design doc requirement).

    Attributes:
        id: Document UUID.
        game_id: UUID of the owning game.
        expansion_id: UUID of the associated expansion (null for base game).
        name: Document display name.
        type: Content classification.
        format: File format.
        status: Processing status.
        current_version: Active version number.
        file_size: File size in bytes.
        error_message: Processing error details (null if none).
        uploaded_at: When the document was first uploaded.
        processed_at: When processing completed (null if not processed).
        updated_at: Last modification timestamp.
        archived_at: Soft-delete timestamp (null if active).
    """

    id: uuid.UUID
    game_id: uuid.UUID
    expansion_id: uuid.UUID | None
    name: str
    type: DocumentTypeFilter
    format: DocumentFormatFilter
    status: DocumentStatusFilter
    current_version: int
    file_size: int
    error_message: str | None
    uploaded_at: datetime
    processed_at: datetime | None
    updated_at: datetime
    archived_at: datetime | None

    model_config = {"from_attributes": True}


class DocumentDetailResponse(DocumentResponse):
    """Response schema for document detail with processing stats.

    Extends DocumentResponse with chunk_count and version_count
    for the detail view.

    Attributes:
        chunk_count: Number of content chunks for this document.
        version_count: Number of versions for this document.
    """

    chunk_count: int
    version_count: int


# ---------------------------------------------------------------------------
# Filter schemas
# ---------------------------------------------------------------------------


class DocumentFilters(BaseModel):
    """Parsed query parameters for the document list endpoint.

    Attributes:
        status: Filter by processing status (comma-separated OR logic).
        type: Filter by document type (comma-separated OR logic).
        expansion_id: Filter by expansion. Use string "null" for base game only.
        sort: Sort field with optional - prefix for descending.
    """

    status: list[DocumentStatusFilter] | None = None
    type: list[DocumentTypeFilter] | None = None
    expansion_id: uuid.UUID | str | None = None
    sort: str = "-uploaded_at"

    @model_validator(mode="before")
    @classmethod
    def parse_comma_separated(cls, data: Any) -> Any:
        """Parse comma-separated strings for status and type fields."""
        if isinstance(data, dict):
            raw_status = data.get("status")
            if isinstance(raw_status, str):
                data["status"] = [v.strip() for v in raw_status.split(",") if v.strip()]
            raw_type = data.get("type")
            if isinstance(raw_type, str):
                data["type"] = [v.strip() for v in raw_type.split(",") if v.strip()]
        return data
