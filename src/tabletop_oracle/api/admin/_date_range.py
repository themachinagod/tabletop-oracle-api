"""Shared date range validation for admin API endpoints.

Centralises the start <= end and max 365 days validation so that
both usage and quality routers validate identically at the API layer
(in addition to service-layer validation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tabletop_oracle.errors.exceptions import ValidationError

if TYPE_CHECKING:
    from datetime import date

_MAX_RANGE_DAYS = 365


def validate_date_range(period_start: date, period_end: date) -> None:
    """Validate that the date range is well-formed.

    Args:
        period_start: Inclusive start date.
        period_end: Inclusive end date.

    Raises:
        ValidationError: If start > end or range exceeds 365 days.
    """
    if period_start > period_end:
        raise ValidationError("period_start must be <= period_end")

    delta = (period_end - period_start).days
    if delta > _MAX_RANGE_DAYS:
        raise ValidationError(
            f"Date range must not exceed {_MAX_RANGE_DAYS} days (requested {delta} days)"
        )
