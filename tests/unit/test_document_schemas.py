"""Unit tests for Document Pydantic schemas.

Tests validation, serialization, and filter parsing for all document
request/response schemas.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from tabletop_oracle.schemas.document import (
    DocumentDetailResponse,
    DocumentExpansionUpdate,
    DocumentFilters,
    DocumentFormatFilter,
    DocumentResponse,
    DocumentStatusFilter,
    DocumentTypeFilter,
    DocumentTypeUpdate,
)

# ---------------------------------------------------------------------------
# DocumentTypeUpdate
# ---------------------------------------------------------------------------


class TestDocumentTypeUpdate:
    """Tests for DocumentTypeUpdate request schema."""

    def test_valid_type(self) -> None:
        """Valid document type is accepted."""
        update = DocumentTypeUpdate(type="core_rules")
        assert update.type == DocumentTypeFilter.CORE_RULES

    def test_all_types_accepted(self) -> None:
        """All enum values are valid."""
        for t in DocumentTypeFilter:
            update = DocumentTypeUpdate(type=t.value)
            assert update.type == t

    def test_invalid_type_rejected(self) -> None:
        """Invalid type value raises validation error."""
        with pytest.raises(ValueError):
            DocumentTypeUpdate(type="not_a_type")


# ---------------------------------------------------------------------------
# DocumentExpansionUpdate
# ---------------------------------------------------------------------------


class TestDocumentExpansionUpdate:
    """Tests for DocumentExpansionUpdate request schema."""

    def test_valid_expansion_id(self) -> None:
        """Valid UUID is accepted."""
        exp_id = uuid.uuid4()
        update = DocumentExpansionUpdate(expansion_id=exp_id)
        assert update.expansion_id == exp_id

    def test_null_expansion_id(self) -> None:
        """Null expansion_id is accepted (base game association)."""
        update = DocumentExpansionUpdate(expansion_id=None)
        assert update.expansion_id is None


# ---------------------------------------------------------------------------
# DocumentResponse
# ---------------------------------------------------------------------------


class TestDocumentResponse:
    """Tests for DocumentResponse schema."""

    def test_from_attributes(self) -> None:
        """Response can be constructed from ORM-like attributes."""
        now = datetime.now(UTC)
        response = DocumentResponse(
            id=uuid.uuid4(),
            game_id=uuid.uuid4(),
            expansion_id=None,
            name="Test Document.pdf",
            type=DocumentTypeFilter.CORE_RULES,
            format=DocumentFormatFilter.PDF,
            status=DocumentStatusFilter.UPLOADED,
            current_version=1,
            file_size=1024,
            error_message=None,
            uploaded_at=now,
            processed_at=None,
            updated_at=now,
            archived_at=None,
        )
        assert response.name == "Test Document.pdf"
        assert response.type == DocumentTypeFilter.CORE_RULES

    def test_serialization_excludes_file_path(self) -> None:
        """Response serialization does not include file_path."""
        now = datetime.now(UTC)
        response = DocumentResponse(
            id=uuid.uuid4(),
            game_id=uuid.uuid4(),
            expansion_id=None,
            name="Test.pdf",
            type=DocumentTypeFilter.FAQ,
            format=DocumentFormatFilter.PDF,
            status=DocumentStatusFilter.PROCESSED,
            current_version=1,
            file_size=2048,
            error_message=None,
            uploaded_at=now,
            processed_at=now,
            updated_at=now,
            archived_at=None,
        )
        data = response.model_dump()
        assert "file_path" not in data


# ---------------------------------------------------------------------------
# DocumentDetailResponse
# ---------------------------------------------------------------------------


class TestDocumentDetailResponse:
    """Tests for DocumentDetailResponse schema."""

    def test_includes_aggregated_counts(self) -> None:
        """Detail response includes chunk_count and version_count."""
        now = datetime.now(UTC)
        response = DocumentDetailResponse(
            id=uuid.uuid4(),
            game_id=uuid.uuid4(),
            expansion_id=None,
            name="Test.pdf",
            type=DocumentTypeFilter.CORE_RULES,
            format=DocumentFormatFilter.PDF,
            status=DocumentStatusFilter.PROCESSED,
            current_version=2,
            file_size=4096,
            error_message=None,
            uploaded_at=now,
            processed_at=now,
            updated_at=now,
            archived_at=None,
            chunk_count=47,
            version_count=2,
        )
        assert response.chunk_count == 47
        assert response.version_count == 2


# ---------------------------------------------------------------------------
# DocumentFilters
# ---------------------------------------------------------------------------


class TestDocumentFilters:
    """Tests for DocumentFilters query parameter parsing."""

    def test_defaults(self) -> None:
        """Default filter values."""
        filters = DocumentFilters()
        assert filters.status is None
        assert filters.type is None
        assert filters.expansion_id is None
        assert filters.sort == "-uploaded_at"

    def test_comma_separated_status(self) -> None:
        """Comma-separated status string is parsed into list."""
        filters = DocumentFilters(status="uploaded,processed")
        assert filters.status is not None
        assert len(filters.status) == 2
        assert DocumentStatusFilter.UPLOADED in filters.status
        assert DocumentStatusFilter.PROCESSED in filters.status

    def test_comma_separated_type(self) -> None:
        """Comma-separated type string is parsed into list."""
        filters = DocumentFilters(type="core_rules,faq")
        assert filters.type is not None
        assert len(filters.type) == 2

    def test_single_status(self) -> None:
        """Single status value is parsed into a single-element list."""
        filters = DocumentFilters(status="error")
        assert filters.status == [DocumentStatusFilter.ERROR]

    def test_expansion_id_uuid(self) -> None:
        """UUID string for expansion_id is accepted."""
        exp_id = uuid.uuid4()
        filters = DocumentFilters(expansion_id=exp_id)
        assert filters.expansion_id == exp_id

    def test_expansion_id_null_string(self) -> None:
        """String 'null' for expansion_id is preserved as string."""
        filters = DocumentFilters(expansion_id="null")
        assert filters.expansion_id == "null"

    def test_custom_sort(self) -> None:
        """Custom sort parameter is accepted."""
        filters = DocumentFilters(sort="name")
        assert filters.sort == "name"
