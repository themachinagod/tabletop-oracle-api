"""Integration tests for pagination, sorting, and filtering against a real database.

Uses testcontainers (or CI-provided PostgreSQL) to verify that the pagination,
sorting, and filtering utilities produce correct SQL and return expected results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy import Boolean, Integer, String, select
from sqlalchemy.orm import Mapped, mapped_column

from tabletop_oracle.api.filtering import (
    FilterOperator,
    FilterSpec,
    SortParam,
    apply_filters,
    apply_sort,
)
from tabletop_oracle.api.pagination import PaginationParams, paginate
from tabletop_oracle.models.base import MappedBase

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class TestItem(MappedBase):
    """Temporary test table for integration tests."""

    __tablename__ = "test_pagination_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    score: Mapped[int] = mapped_column(Integer, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


@pytest.fixture
async def _create_test_table(db_session: AsyncSession) -> None:
    """Create the test table and seed data."""
    conn = await db_session.connection()
    await conn.run_sync(MappedBase.metadata.create_all, tables=[TestItem.__table__])

    items = [
        TestItem(name="Alpha", score=10, active=True),
        TestItem(name="Bravo", score=20, active=True),
        TestItem(name="Charlie", score=30, active=False),
        TestItem(name="Delta", score=40, active=True),
        TestItem(name="Echo", score=50, active=False),
    ]
    db_session.add_all(items)
    await db_session.flush()


@pytest.mark.integration
class TestPaginateIntegration:
    """Integration tests for paginate() against a real database."""

    @pytest.mark.usefixtures("_create_test_table")
    async def test_paginate_first_page(self, db_session: AsyncSession) -> None:
        """First page returns correct items and total count."""
        query = select(TestItem)
        params = PaginationParams(page=1, page_size=2)
        items, total = await paginate(db_session, query, params)

        assert total == 5
        assert len(items) == 2

    @pytest.mark.usefixtures("_create_test_table")
    async def test_paginate_last_page_partial(self, db_session: AsyncSession) -> None:
        """Last page returns remaining items when not full."""
        query = select(TestItem)
        params = PaginationParams(page=3, page_size=2)
        items, total = await paginate(db_session, query, params)

        assert total == 5
        assert len(items) == 1

    @pytest.mark.usefixtures("_create_test_table")
    async def test_paginate_beyond_total(self, db_session: AsyncSession) -> None:
        """Page beyond total returns empty list with correct count."""
        query = select(TestItem)
        params = PaginationParams(page=10, page_size=25)
        items, total = await paginate(db_session, query, params)

        assert total == 5
        assert len(items) == 0

    @pytest.mark.usefixtures("_create_test_table")
    async def test_paginate_empty_result(self, db_session: AsyncSession) -> None:
        """Paginating a query with no matches returns 0 count and empty list."""
        query = select(TestItem).where(TestItem.score > 9999)
        params = PaginationParams(page=1, page_size=25)
        items, total = await paginate(db_session, query, params)

        assert total == 0
        assert len(items) == 0


@pytest.mark.integration
class TestSortIntegration:
    """Integration tests for apply_sort() against a real database."""

    @pytest.mark.usefixtures("_create_test_table")
    async def test_sort_ascending(self, db_session: AsyncSession) -> None:
        """Ascending sort returns items in correct order."""
        query = select(TestItem)
        sort = SortParam(field_name="name", descending=False)
        sorted_query = apply_sort(query, sort, TestItem)

        result = await db_session.execute(sorted_query)
        items = list(result.scalars().all())

        assert items[0].name == "Alpha"
        assert items[-1].name == "Echo"

    @pytest.mark.usefixtures("_create_test_table")
    async def test_sort_descending(self, db_session: AsyncSession) -> None:
        """Descending sort returns items in reverse order."""
        query = select(TestItem)
        sort = SortParam(field_name="score", descending=True)
        sorted_query = apply_sort(query, sort, TestItem)

        result = await db_session.execute(sorted_query)
        items = list(result.scalars().all())

        assert items[0].score == 50
        assert items[-1].score == 10


@pytest.mark.integration
class TestFilterIntegration:
    """Integration tests for apply_filters() against a real database."""

    @pytest.mark.usefixtures("_create_test_table")
    async def test_equality_filter(self, db_session: AsyncSession) -> None:
        """EQ filter returns matching items."""
        query = select(TestItem)
        filters = [FilterSpec(field_name="name", operator=FilterOperator.EQ, value="Alpha")]
        filtered = apply_filters(query, filters, TestItem)

        result = await db_session.execute(filtered)
        items = list(result.scalars().all())

        assert len(items) == 1
        assert items[0].name == "Alpha"

    @pytest.mark.usefixtures("_create_test_table")
    async def test_range_filter(self, db_session: AsyncSession) -> None:
        """GTE + LTE range filter returns items within range."""
        query = select(TestItem)
        filters = [
            FilterSpec(field_name="score", operator=FilterOperator.GTE, value=20),
            FilterSpec(field_name="score", operator=FilterOperator.LTE, value=40),
        ]
        filtered = apply_filters(query, filters, TestItem)

        result = await db_session.execute(filtered)
        items = list(result.scalars().all())

        assert len(items) == 3
        scores = {item.score for item in items}
        assert scores == {20, 30, 40}

    @pytest.mark.usefixtures("_create_test_table")
    async def test_search_filter(self, db_session: AsyncSession) -> None:
        """SEARCH filter matches via ILIKE."""
        query = select(TestItem)
        filters = [FilterSpec(field_name="name", operator=FilterOperator.SEARCH, value="cha")]
        filtered = apply_filters(query, filters, TestItem)

        result = await db_session.execute(filtered)
        items = list(result.scalars().all())

        assert len(items) == 1
        assert items[0].name == "Charlie"

    @pytest.mark.usefixtures("_create_test_table")
    async def test_bool_filter(self, db_session: AsyncSession) -> None:
        """BOOL filter matches boolean column."""
        query = select(TestItem)
        filters = [FilterSpec(field_name="active", operator=FilterOperator.BOOL, value=False)]
        filtered = apply_filters(query, filters, TestItem)

        result = await db_session.execute(filtered)
        items = list(result.scalars().all())

        assert len(items) == 2
        names = {item.name for item in items}
        assert names == {"Charlie", "Echo"}

    @pytest.mark.usefixtures("_create_test_table")
    async def test_in_filter(self, db_session: AsyncSession) -> None:
        """IN filter with multiple values returns matching items."""
        query = select(TestItem)
        filters = [
            FilterSpec(
                field_name="name",
                operator=FilterOperator.IN,
                value=["Alpha", "Delta"],
            )
        ]
        filtered = apply_filters(query, filters, TestItem)

        result = await db_session.execute(filtered)
        items = list(result.scalars().all())

        assert len(items) == 2
        names = {item.name for item in items}
        assert names == {"Alpha", "Delta"}


@pytest.mark.integration
class TestCombinedIntegration:
    """Integration tests combining pagination, sorting, and filtering."""

    @pytest.mark.usefixtures("_create_test_table")
    async def test_filter_sort_paginate(self, db_session: AsyncSession) -> None:
        """Combined filter + sort + paginate returns correct results."""
        query = select(TestItem)
        filters = [FilterSpec(field_name="active", operator=FilterOperator.BOOL, value=True)]
        query = apply_filters(query, filters, TestItem)
        sort = SortParam(field_name="score", descending=True)
        query = apply_sort(query, sort, TestItem)

        params = PaginationParams(page=1, page_size=2)
        items, total = await paginate(db_session, query, params)

        # 3 active items total: Delta(40), Bravo(20), Alpha(10)
        assert total == 3
        assert len(items) == 2
        assert items[0].score == 40  # Delta
        assert items[1].score == 20  # Bravo
