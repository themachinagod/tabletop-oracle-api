"""Pagination dependency and query helpers.

Provides a FastAPI dependency for parsing pagination query parameters
and utilities for applying OFFSET/LIMIT to SQLAlchemy queries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fastapi import Query
from sqlalchemy import func, select

if TYPE_CHECKING:
    from sqlalchemy import Select
    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class PaginationParams:
    """Parsed pagination query parameters.

    Attributes:
        page: 1-indexed page number (minimum 1).
        page_size: Items per page (minimum 1, maximum 100).
    """

    page: int
    page_size: int

    @property
    def offset(self) -> int:
        """Calculate the SQL OFFSET for the current page."""
        return (self.page - 1) * self.page_size


def get_pagination_params(
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(default=25, ge=1, le=100, description="Items per page (max 100)"),
) -> PaginationParams:
    """FastAPI dependency that parses and validates pagination query parameters.

    Args:
        page: Page number, 1-indexed, minimum 1.
        page_size: Items per page, minimum 1, maximum 100.

    Returns:
        Validated PaginationParams instance.
    """
    return PaginationParams(page=page, page_size=page_size)


async def paginate(
    session: AsyncSession,
    query: Select[Any],
    params: PaginationParams,
) -> tuple[list[Any], int]:
    """Apply pagination to a SQLAlchemy select query.

    Executes two queries: a COUNT for total results, and the paginated
    SELECT with OFFSET/LIMIT applied.

    Args:
        session: Async database session.
        query: Base SQLAlchemy SELECT statement (without OFFSET/LIMIT).
        params: Pagination parameters.

    Returns:
        Tuple of (items, total_count).
    """
    count_query = select(func.count()).select_from(query.subquery())
    count_result = await session.execute(count_query)
    total_count = count_result.scalar_one()

    paginated_query = query.offset(params.offset).limit(params.page_size)
    result = await session.execute(paginated_query)
    items: list[Any] = list(result.scalars().all())

    return items, total_count


def compute_total_pages(total_count: int, page_size: int) -> int:
    """Calculate the total number of pages.

    Args:
        total_count: Total number of items.
        page_size: Items per page.

    Returns:
        Total pages (0 when there are no items).
    """
    if total_count == 0:
        return 0
    return math.ceil(total_count / page_size)
