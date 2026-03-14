"""Unit tests for response builder helpers."""

from unittest.mock import MagicMock

from tabletop_oracle.api.pagination import PaginationParams
from tabletop_oracle.api.response import ok, ok_list


def _make_request(request_id: str = "test-req-id") -> MagicMock:
    """Create a mock Starlette Request with request_id on state."""
    request = MagicMock()
    request.state.request_id = request_id
    return request


class TestOk:
    """Tests for the ok() response helper."""

    def test_wraps_data_in_envelope(self) -> None:
        """ok() wraps data in a DataEnvelope."""
        request = _make_request("req-123")
        result = ok({"name": "Catan"}, request)
        assert result.data == {"name": "Catan"}

    def test_includes_request_id_in_meta(self) -> None:
        """ok() populates meta.request_id from request state."""
        request = _make_request("abc-def-123")
        result = ok("payload", request)
        assert result.meta.request_id == "abc-def-123"

    def test_serialization_shape(self) -> None:
        """ok() produces the correct JSON shape."""
        request = _make_request("req-456")
        result = ok(42, request)
        dumped = result.model_dump()
        assert dumped == {
            "data": 42,
            "meta": {"request_id": "req-456"},
        }


class TestOkList:
    """Tests for the ok_list() response helper."""

    def test_wraps_items_in_list_envelope(self) -> None:
        """ok_list() wraps items in a ListEnvelope."""
        request = _make_request("req-789")
        params = PaginationParams(page=1, page_size=25)
        result = ok_list(["a", "b"], 2, params, request)
        assert result.data == ["a", "b"]

    def test_includes_pagination_metadata(self) -> None:
        """ok_list() includes correct pagination metadata."""
        request = _make_request("req-pag")
        params = PaginationParams(page=2, page_size=10)
        result = ok_list(["x"], 25, params, request)
        assert result.meta.pagination.page == 2
        assert result.meta.pagination.page_size == 10
        assert result.meta.pagination.total_count == 25
        assert result.meta.pagination.total_pages == 3

    def test_includes_request_id(self) -> None:
        """ok_list() populates meta.request_id."""
        request = _make_request("list-req-id")
        params = PaginationParams(page=1, page_size=10)
        result = ok_list([], 0, params, request)
        assert result.meta.request_id == "list-req-id"

    def test_empty_results(self) -> None:
        """ok_list() handles empty results correctly."""
        request = _make_request("empty-req")
        params = PaginationParams(page=1, page_size=25)
        result = ok_list([], 0, params, request)
        assert result.data == []
        assert result.meta.pagination.total_count == 0
        assert result.meta.pagination.total_pages == 0

    def test_serialization_shape(self) -> None:
        """ok_list() produces the correct JSON shape."""
        request = _make_request("shape-req")
        params = PaginationParams(page=1, page_size=25)
        result = ok_list(["item1"], 1, params, request)
        dumped = result.model_dump()
        assert dumped == {
            "data": ["item1"],
            "meta": {
                "request_id": "shape-req",
                "pagination": {
                    "page": 1,
                    "page_size": 25,
                    "total_count": 1,
                    "total_pages": 1,
                },
            },
        }

    def test_page_beyond_total_returns_empty_data(self) -> None:
        """ok_list() with page beyond total returns empty data with correct meta."""
        request = _make_request("beyond-req")
        params = PaginationParams(page=5, page_size=25)
        result = ok_list([], 50, params, request)
        assert result.data == []
        assert result.meta.pagination.page == 5
        assert result.meta.pagination.total_count == 50
        assert result.meta.pagination.total_pages == 2
