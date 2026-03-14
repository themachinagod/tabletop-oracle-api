"""Unit tests for common response schemas."""

from tabletop_oracle.schemas.common import (
    DataEnvelope,
    ErrorDetail,
    ErrorEnvelope,
    ErrorMeta,
    FieldErrorDetail,
    ListEnvelope,
    PaginationMeta,
)


def test_pagination_meta() -> None:
    """PaginationMeta stores page info."""
    meta = PaginationMeta(page=1, page_size=20, total_items=50, total_pages=3)
    assert meta.page == 1
    assert meta.total_pages == 3


def test_data_envelope() -> None:
    """DataEnvelope wraps arbitrary data."""
    envelope = DataEnvelope[str](data="hello")
    assert envelope.data == "hello"


def test_list_envelope() -> None:
    """ListEnvelope wraps a list with pagination metadata."""
    meta = PaginationMeta(page=1, page_size=10, total_items=2, total_pages=1)
    envelope = ListEnvelope[str](data=["a", "b"], meta=meta)
    assert len(envelope.data) == 2
    assert envelope.meta.total_items == 2


def test_field_error_detail() -> None:
    """FieldErrorDetail stores field, message, and code."""
    detail = FieldErrorDetail(field="email", message="invalid")
    assert detail.code == "invalid"

    detail_custom = FieldErrorDetail(field="age", message="too low", code="min_value")
    assert detail_custom.code == "min_value"


def test_error_detail_without_details() -> None:
    """ErrorDetail defaults to empty details list."""
    detail = ErrorDetail(code="NOT_FOUND", message="not found")
    assert detail.details == []


def test_error_detail_with_details() -> None:
    """ErrorDetail carries per-field validation details."""
    field_errors = [
        FieldErrorDetail(field="name", message="required", code="required"),
    ]
    detail = ErrorDetail(code="VALIDATION_ERROR", message="bad input", details=field_errors)
    assert len(detail.details) == 1
    assert detail.details[0].field == "name"


def test_error_meta() -> None:
    """ErrorMeta stores request_id."""
    meta = ErrorMeta(request_id="abc-123")
    assert meta.request_id == "abc-123"


def test_error_envelope() -> None:
    """ErrorEnvelope wraps ErrorDetail and ErrorMeta."""
    envelope = ErrorEnvelope(
        error=ErrorDetail(code="ERR", message="fail"),
        meta=ErrorMeta(request_id="req-1"),
    )
    assert envelope.error.code == "ERR"
    assert envelope.meta.request_id == "req-1"
