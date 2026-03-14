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


class ErrorDetail(BaseModel):
    """Structured error detail."""

    code: str
    message: str
    field: str | None = None


class ErrorEnvelope(BaseModel):
    """Standard error response envelope."""

    error: ErrorDetail
