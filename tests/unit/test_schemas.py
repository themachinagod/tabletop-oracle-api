"""Unit tests for common response schemas."""

from tabletop_oracle.schemas.common import (
    DataEnvelope,
    ErrorDetail,
    ErrorEnvelope,
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


def test_error_detail() -> None:
    """ErrorDetail stores code, message, and optional field."""
    detail = ErrorDetail(code="NOT_FOUND", message="not found")
    assert detail.field is None

    detail_with_field = ErrorDetail(code="VALIDATION", message="bad", field="name")
    assert detail_with_field.field == "name"


def test_error_envelope() -> None:
    """ErrorEnvelope wraps an ErrorDetail."""
    envelope = ErrorEnvelope(error=ErrorDetail(code="ERR", message="fail"))
    assert envelope.error.code == "ERR"
