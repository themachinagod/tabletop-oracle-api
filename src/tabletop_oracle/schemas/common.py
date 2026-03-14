"""Common response schemas: envelope, pagination, errors.

Defines the standard API response envelopes per F001 design:
- DataEnvelope[T]: single-resource responses with meta.request_id
- ListEnvelope[T]: collection responses with pagination metadata
- ErrorEnvelope: structured error responses
"""

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class PaginationMeta(BaseModel):
    """Pagination metadata for list responses.

    Uses total_count (not total_items) per F001 design spec.
    """

    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=100)
    total_count: int = Field(ge=0)
    total_pages: int = Field(ge=0)


class ResponseMeta(BaseModel):
    """Success response metadata containing the correlation request ID."""

    request_id: str


class ListResponseMeta(BaseModel):
    """List response metadata with request ID and pagination info."""

    request_id: str
    pagination: PaginationMeta


class DataEnvelope(BaseModel, Generic[T]):
    """Standard response envelope wrapping a single resource.

    Shape: ``{ "data": T, "meta": { "request_id": "uuid" } }``
    """

    data: T
    meta: ResponseMeta


class ListEnvelope(BaseModel, Generic[T]):
    """Standard response envelope for paginated lists.

    Shape: ``{ "data": [T], "meta": { "request_id": "uuid", "pagination": {...} } }``
    """

    data: list[T]
    meta: ListResponseMeta


class FieldErrorDetail(BaseModel):
    """Per-field validation error detail."""

    field: str
    message: str
    code: str = "invalid"


class ErrorDetail(BaseModel):
    """Structured error detail.

    The details array is populated for validation errors with per-field
    information. For non-validation errors, details is empty.
    """

    code: str
    message: str
    details: list[FieldErrorDetail] = Field(default_factory=list)


class ErrorMeta(BaseModel):
    """Error response metadata."""

    request_id: str


class ErrorEnvelope(BaseModel):
    """Standard error response envelope.

    All error responses follow this structure. The meta.request_id
    is populated from the correlation middleware.
    """

    error: ErrorDetail
    meta: ErrorMeta
