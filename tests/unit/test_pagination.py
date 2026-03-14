"""Unit tests for pagination utilities."""

import pytest

from tabletop_oracle.api.pagination import (
    PaginationParams,
    compute_total_pages,
    get_pagination_params,
)


class TestPaginationParams:
    """Tests for PaginationParams dataclass."""

    def test_offset_first_page(self) -> None:
        """Page 1 produces offset 0."""
        params = PaginationParams(page=1, page_size=25)
        assert params.offset == 0

    def test_offset_second_page(self) -> None:
        """Page 2 with page_size 25 produces offset 25."""
        params = PaginationParams(page=2, page_size=25)
        assert params.offset == 25

    def test_offset_custom_page_size(self) -> None:
        """Offset calculation respects custom page_size."""
        params = PaginationParams(page=3, page_size=10)
        assert params.offset == 20

    def test_offset_large_page(self) -> None:
        """Large page number produces correct offset."""
        params = PaginationParams(page=100, page_size=50)
        assert params.offset == 4950


class TestGetPaginationParams:
    """Tests for the FastAPI dependency function."""

    def test_with_explicit_values(self) -> None:
        """Accepts explicit page and page_size values.

        Note: get_pagination_params is designed as a FastAPI dependency.
        FastAPI resolves Query() defaults before calling the function.
        Direct calls must provide explicit integer arguments.
        """
        params = get_pagination_params(page=3, page_size=50)
        assert params.page == 3
        assert params.page_size == 50

    def test_returns_pagination_params(self) -> None:
        """Returns a PaginationParams instance."""
        params = get_pagination_params(page=1, page_size=25)
        assert isinstance(params, PaginationParams)
        assert params.page == 1
        assert params.page_size == 25

    def test_minimum_page(self) -> None:
        """page=1 is the minimum valid value."""
        params = get_pagination_params(page=1, page_size=25)
        assert params.page == 1

    def test_maximum_page_size(self) -> None:
        """page_size=100 is the maximum valid value."""
        params = get_pagination_params(page=1, page_size=100)
        assert params.page_size == 100


class TestComputeTotalPages:
    """Tests for total pages calculation."""

    def test_exact_division(self) -> None:
        """100 items / 25 per page = 4 pages."""
        assert compute_total_pages(100, 25) == 4

    def test_remainder_adds_page(self) -> None:
        """101 items / 25 per page = 5 pages (rounds up)."""
        assert compute_total_pages(101, 25) == 5

    def test_single_item(self) -> None:
        """1 item always needs 1 page."""
        assert compute_total_pages(1, 25) == 1

    def test_zero_items(self) -> None:
        """0 items = 0 pages."""
        assert compute_total_pages(0, 25) == 0

    def test_items_less_than_page_size(self) -> None:
        """Fewer items than page_size = 1 page."""
        assert compute_total_pages(10, 25) == 1

    def test_page_size_one(self) -> None:
        """page_size=1 means total_pages equals total_count."""
        assert compute_total_pages(5, 1) == 5

    @pytest.mark.parametrize(
        ("total_count", "page_size", "expected"),
        [
            (0, 10, 0),
            (1, 10, 1),
            (10, 10, 1),
            (11, 10, 2),
            (99, 100, 1),
            (100, 100, 1),
            (101, 100, 2),
        ],
    )
    def test_boundary_cases(self, total_count: int, page_size: int, expected: int) -> None:
        """Boundary conditions for page calculation."""
        assert compute_total_pages(total_count, page_size) == expected
