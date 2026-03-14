"""Filtering and sorting utilities for collection endpoints.

Provides reusable components for parsing sort/filter query parameters
and applying them as SQLAlchemy WHERE and ORDER BY clauses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException
from sqlalchemy import Select, or_

if TYPE_CHECKING:
    from sqlalchemy.orm import DeclarativeBase


class FilterOperator(StrEnum):
    """Supported filter operators."""

    EQ = "eq"
    GTE = "gte"
    LTE = "lte"
    GT = "gt"
    LT = "lt"
    IN = "in"
    SEARCH = "search"
    BOOL = "bool"


@dataclass(frozen=True)
class SortParam:
    """Parsed sort parameter.

    Attributes:
        field_name: Column name to sort by.
        descending: True for descending order, False for ascending.
    """

    field_name: str
    descending: bool = False


@dataclass(frozen=True)
class FilterSpec:
    """A single filter specification.

    Attributes:
        field_name: Column name to filter on.
        operator: The filter operator to apply.
        value: The filter value(s).
    """

    field_name: str
    operator: FilterOperator
    value: Any


@dataclass
class FilterParam:
    """Collection of filter specifications for a query.

    Declare allowed filterable fields, then parse incoming query params
    to build filter specs. Subclass or instantiate directly.

    Attributes:
        filters: List of parsed filter specifications.
    """

    filters: list[FilterSpec] = field(default_factory=list)


def parse_sort(
    sort_value: str | None,
    allowed_fields: set[str],
) -> SortParam | None:
    """Parse a sort query parameter string into a SortParam.

    Format: ``field`` for ascending, ``-field`` for descending.
    Raises HTTP 400 if the field is not in the allowed set.

    Args:
        sort_value: Raw sort query parameter (e.g. ``"name"`` or ``"-updated_at"``).
        allowed_fields: Set of column names that are valid sort targets.

    Returns:
        Parsed SortParam, or None if sort_value is None/empty.

    Raises:
        HTTPException: 400 if the sort field is not allowed.
    """
    if not sort_value:
        return None

    descending = sort_value.startswith("-")
    field_name = sort_value.lstrip("-")

    if field_name not in allowed_fields:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sort field '{field_name}'. Allowed: {sorted(allowed_fields)}",
        )

    return SortParam(field_name=field_name, descending=descending)


def apply_sort(
    query: Select[Any],
    sort_param: SortParam | None,
    model: type[DeclarativeBase],
) -> Select[Any]:
    """Apply ORDER BY to a SQLAlchemy query based on a SortParam.

    Args:
        query: The SQLAlchemy SELECT statement.
        sort_param: Parsed sort specification, or None for no sorting.
        model: The SQLAlchemy model class to resolve column references.

    Returns:
        Query with ORDER BY applied (or unchanged if sort_param is None).
    """
    if sort_param is None:
        return query

    column = getattr(model, sort_param.field_name)
    if sort_param.descending:
        return query.order_by(column.desc())
    return query.order_by(column.asc())


def parse_filter_value(raw: str, operator: FilterOperator) -> Any:
    """Convert a raw string filter value to the appropriate Python type.

    Args:
        raw: Raw string value from query parameter.
        operator: The filter operator, which determines parsing behaviour.

    Returns:
        Parsed value appropriate for the operator.
    """
    if operator == FilterOperator.BOOL:
        return raw.lower() in ("true", "1", "yes")
    if operator == FilterOperator.IN:
        return [v.strip() for v in raw.split(",") if v.strip()]
    return raw


def build_filter_spec(
    field_name: str,
    raw_value: str,
    operator: FilterOperator,
) -> FilterSpec:
    """Create a FilterSpec from raw query parameter input.

    Args:
        field_name: The column name to filter on.
        raw_value: Raw string value from the query parameter.
        operator: The filter operator to apply.

    Returns:
        A FilterSpec with the parsed value.
    """
    parsed = parse_filter_value(raw_value, operator)
    return FilterSpec(field_name=field_name, operator=operator, value=parsed)


def apply_filters(
    query: Select[Any],
    filters: list[FilterSpec],
    model: type[DeclarativeBase],
) -> Select[Any]:
    """Apply WHERE clauses to a SQLAlchemy query based on filter specs.

    Supports equality, multi-value OR (IN), range comparisons, text
    search (ILIKE), and boolean filters.

    Args:
        query: The SQLAlchemy SELECT statement.
        filters: List of filter specifications to apply.
        model: The SQLAlchemy model class to resolve column references.

    Returns:
        Query with WHERE clauses applied.
    """
    for spec in filters:
        column = getattr(model, spec.field_name)

        if spec.operator == FilterOperator.EQ:
            query = query.where(column == spec.value)

        elif spec.operator == FilterOperator.IN:
            values = spec.value if isinstance(spec.value, list) else [spec.value]
            if len(values) == 1:
                query = query.where(column == values[0])
            else:
                query = query.where(or_(*[column == v for v in values]))

        elif spec.operator == FilterOperator.GTE:
            query = query.where(column >= spec.value)

        elif spec.operator == FilterOperator.LTE:
            query = query.where(column <= spec.value)

        elif spec.operator == FilterOperator.GT:
            query = query.where(column > spec.value)

        elif spec.operator == FilterOperator.LT:
            query = query.where(column < spec.value)

        elif spec.operator == FilterOperator.SEARCH:
            query = query.where(column.ilike(f"%{spec.value}%"))

        elif spec.operator == FilterOperator.BOOL:
            query = query.where(column.is_(spec.value))

    return query
