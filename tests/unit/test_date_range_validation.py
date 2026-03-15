"""Unit tests for the shared date range validation utility."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from tabletop_oracle.api.admin._date_range import validate_date_range
from tabletop_oracle.errors.exceptions import ValidationError


class TestValidateDateRange:
    """Tests for validate_date_range."""

    def test_valid_range_does_not_raise(self) -> None:
        """A valid date range passes."""
        validate_date_range(date(2026, 1, 1), date(2026, 1, 31))

    def test_same_day_range_is_valid(self) -> None:
        """period_start == period_end is valid."""
        validate_date_range(date(2026, 3, 15), date(2026, 3, 15))

    def test_start_after_end_raises(self) -> None:
        """period_start > period_end raises ValidationError."""
        with pytest.raises(ValidationError, match="period_start must be <= period_end"):
            validate_date_range(date(2026, 2, 1), date(2026, 1, 1))

    def test_range_exactly_365_days_is_valid(self) -> None:
        """Exactly 365 days is the maximum allowed range."""
        start = date(2025, 3, 15)
        end = start + timedelta(days=365)
        validate_date_range(start, end)

    def test_range_exceeds_365_days_raises(self) -> None:
        """Range > 365 days raises ValidationError."""
        start = date(2025, 3, 15)
        end = start + timedelta(days=366)
        with pytest.raises(ValidationError, match="must not exceed 365 days"):
            validate_date_range(start, end)
