"""Unit tests for common response schemas."""

import pytest
from pydantic import ValidationError

from tabletop_oracle.schemas.common import (
    DataEnvelope,
    ErrorDetail,
    ErrorEnvelope,
    ErrorMeta,
    FieldErrorDetail,
    ListEnvelope,
    ListResponseMeta,
    PaginationMeta,
    ResponseMeta,
)


class TestPaginationMeta:
    """Tests for PaginationMeta schema."""

    def test_valid_pagination_meta(self) -> None:
        """PaginationMeta stores page info with total_count field."""
        meta = PaginationMeta(page=1, page_size=25, total_count=142, total_pages=6)
        assert meta.page == 1
        assert meta.page_size == 25
        assert meta.total_count == 142
        assert meta.total_pages == 6

    def test_page_minimum_validation(self) -> None:
        """PaginationMeta rejects page < 1."""
        with pytest.raises(ValidationError):
            PaginationMeta(page=0, page_size=25, total_count=10, total_pages=1)

    def test_page_size_maximum_validation(self) -> None:
        """PaginationMeta rejects page_size > 100."""
        with pytest.raises(ValidationError):
            PaginationMeta(page=1, page_size=101, total_count=10, total_pages=1)

    def test_total_count_non_negative(self) -> None:
        """PaginationMeta rejects negative total_count."""
        with pytest.raises(ValidationError):
            PaginationMeta(page=1, page_size=25, total_count=-1, total_pages=0)


class TestResponseMeta:
    """Tests for ResponseMeta schema."""

    def test_response_meta_stores_request_id(self) -> None:
        """ResponseMeta holds the correlation request_id."""
        meta = ResponseMeta(request_id="abc-123-def")
        assert meta.request_id == "abc-123-def"


class TestDataEnvelope:
    """Tests for DataEnvelope schema."""

    def test_wraps_data_with_meta(self) -> None:
        """DataEnvelope wraps data and includes meta with request_id."""
        envelope = DataEnvelope(
            data={"name": "Catan"},
            meta=ResponseMeta(request_id="req-1"),
        )
        assert envelope.data == {"name": "Catan"}
        assert envelope.meta.request_id == "req-1"

    def test_serialization_shape(self) -> None:
        """DataEnvelope serializes to the expected JSON shape."""
        envelope = DataEnvelope(
            data="hello",
            meta=ResponseMeta(request_id="req-2"),
        )
        dumped = envelope.model_dump()
        assert "data" in dumped
        assert "meta" in dumped
        assert dumped["meta"]["request_id"] == "req-2"


class TestListEnvelope:
    """Tests for ListEnvelope schema."""

    def test_wraps_list_with_pagination_meta(self) -> None:
        """ListEnvelope wraps a list with pagination and request_id metadata."""
        pagination = PaginationMeta(page=1, page_size=10, total_count=2, total_pages=1)
        meta = ListResponseMeta(request_id="req-3", pagination=pagination)
        envelope = ListEnvelope(data=["a", "b"], meta=meta)
        assert len(envelope.data) == 2
        assert envelope.meta.request_id == "req-3"
        assert envelope.meta.pagination.total_count == 2

    def test_serialization_shape(self) -> None:
        """ListEnvelope serializes with nested pagination in meta."""
        pagination = PaginationMeta(page=2, page_size=25, total_count=50, total_pages=2)
        meta = ListResponseMeta(request_id="req-4", pagination=pagination)
        envelope = ListEnvelope(data=[], meta=meta)
        dumped = envelope.model_dump()
        assert dumped["meta"]["request_id"] == "req-4"
        assert dumped["meta"]["pagination"]["total_count"] == 50
        assert dumped["meta"]["pagination"]["page"] == 2


class TestFieldErrorDetail:
    """Tests for FieldErrorDetail schema."""

    def test_default_code(self) -> None:
        """FieldErrorDetail defaults code to 'invalid'."""
        detail = FieldErrorDetail(field="email", message="invalid format")
        assert detail.code == "invalid"

    def test_custom_code(self) -> None:
        """FieldErrorDetail accepts custom code."""
        detail = FieldErrorDetail(field="age", message="too low", code="min_value")
        assert detail.code == "min_value"


class TestErrorDetail:
    """Tests for ErrorDetail schema."""

    def test_empty_details_by_default(self) -> None:
        """ErrorDetail defaults to empty details list."""
        detail = ErrorDetail(code="NOT_FOUND", message="not found")
        assert detail.details == []

    def test_with_field_errors(self) -> None:
        """ErrorDetail carries per-field validation details."""
        field_errors = [
            FieldErrorDetail(field="name", message="required", code="required"),
        ]
        detail = ErrorDetail(code="VALIDATION_ERROR", message="bad input", details=field_errors)
        assert len(detail.details) == 1
        assert detail.details[0].field == "name"


class TestErrorEnvelope:
    """Tests for ErrorEnvelope schema."""

    def test_wraps_error_and_meta(self) -> None:
        """ErrorEnvelope wraps ErrorDetail and ErrorMeta."""
        envelope = ErrorEnvelope(
            error=ErrorDetail(code="ERR", message="fail"),
            meta=ErrorMeta(request_id="req-1"),
        )
        assert envelope.error.code == "ERR"
        assert envelope.meta.request_id == "req-1"
