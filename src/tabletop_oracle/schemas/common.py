"""Common response schemas: envelope, pagination, errors."""

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class PaginationMeta(BaseModel):
    """Pagination metadata for list responses."""

    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=100)
    total_items: int = Field(ge=0)
    total_pages: int = Field(ge=0)


class DataEnvelope(BaseModel, Generic[T]):
    """Standard response envelope wrapping data."""

    data: T


class ListEnvelope(BaseModel, Generic[T]):
    """Standard response envelope for paginated lists."""

    data: list[T]
    meta: PaginationMeta


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
