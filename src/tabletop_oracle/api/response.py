"""Response builder helpers.

Convenience functions for constructing standard API response envelopes.
Reads request_id from ``request.state.request_id`` (set by CorrelationMiddleware).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tabletop_oracle.api.pagination import PaginationParams, compute_total_pages
from tabletop_oracle.schemas.common import (
    DataEnvelope,
    ListEnvelope,
    ListResponseMeta,
    PaginationMeta,
    ResponseMeta,
)

if TYPE_CHECKING:
    from starlette.requests import Request


def ok(data: object, request: Request) -> DataEnvelope[object]:
    """Build a DataEnvelope response for a single resource.

    Args:
        data: The response payload.
        request: The Starlette/FastAPI request (provides request_id via state).

    Returns:
        DataEnvelope with data and meta.request_id.
    """
    request_id: str = request.state.request_id
    return DataEnvelope(
        data=data,
        meta=ResponseMeta(request_id=request_id),
    )


def ok_list(
    items: list[object],
    total_count: int,
    params: PaginationParams,
    request: Request,
) -> ListEnvelope[object]:
    """Build a ListEnvelope response for a paginated collection.

    Args:
        items: The list of response items.
        total_count: Total number of items matching the query (pre-pagination).
        params: The pagination parameters used for the query.
        request: The Starlette/FastAPI request (provides request_id via state).

    Returns:
        ListEnvelope with data, pagination metadata, and meta.request_id.
    """
    request_id: str = request.state.request_id
    total_pages = compute_total_pages(total_count, params.page_size)

    return ListEnvelope(
        data=items,
        meta=ListResponseMeta(
            request_id=request_id,
            pagination=PaginationMeta(
                page=params.page,
                page_size=params.page_size,
                total_count=total_count,
                total_pages=total_pages,
            ),
        ),
    )
