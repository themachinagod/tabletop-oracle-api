"""Unit tests for UsageAggregationService.

Tests all aggregation methods (get_summary, get_trends, get_by_capability,
get_by_game, get_by_user, get_guardrail_status), date range validation,
and guardrail threshold computation. Uses mock database sessions with
controlled return values rather than a real database.
"""

from __future__ import annotations

import uuid
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from tabletop_oracle.errors.exceptions import ValidationError
from tabletop_oracle.models.guardrail import GuardrailConfig
from tabletop_oracle.schemas.usage import (
    CapabilityUsageResponse,
    DailyUsageResponse,
    GameUsageResponse,
    GuardrailStatusResponse,
    GuardrailThreshold,
    UsageSummaryResponse,
    UserUsageResponse,
)
from tabletop_oracle.services.admin.usage_aggregation import UsageAggregationService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_db_returning_one(row: tuple[object, ...]) -> AsyncMock:
    """Build a mock AsyncSession that returns a single row from execute."""
    result_mock = MagicMock()
    result_mock.one.return_value = row
    result_mock.scalar_one.return_value = row[0]
    result_mock.scalar_one_or_none.return_value = None
    result_mock.all.return_value = [row]

    db = AsyncMock()
    db.execute = AsyncMock(return_value=result_mock)
    return db


def _mock_db_returning_rows(rows: list[tuple[object, ...]]) -> AsyncMock:
    """Build a mock AsyncSession that returns multiple rows from execute."""
    result_mock = MagicMock()
    result_mock.all.return_value = rows

    db = AsyncMock()
    db.execute = AsyncMock(return_value=result_mock)
    return db


def _mock_db_for_guardrail(
    config: GuardrailConfig | None,
    daily_tokens: int = 0,
    daily_queries: int = 0,
) -> AsyncMock:
    """Build a mock that handles the guardrail status query sequence.

    The service calls execute 3 times:
    1. Load guardrail config (scalar_one_or_none)
    2. Get daily token total (scalar_one)
    3. Get daily query count (scalar_one)
    """
    config_result = MagicMock()
    config_result.scalar_one_or_none.return_value = config

    token_result = MagicMock()
    token_result.scalar_one.return_value = daily_tokens

    query_result = MagicMock()
    query_result.scalar_one.return_value = daily_queries

    db = AsyncMock()
    db.execute = AsyncMock(side_effect=[config_result, token_result, query_result])
    return db


def _make_guardrail_config(
    *,
    enforcement_enabled: bool = True,
    daily_token_budget: int | None = None,
    daily_query_budget: int | None = None,
) -> GuardrailConfig:
    """Build a GuardrailConfig mock with specified values."""
    config = MagicMock(spec=GuardrailConfig)
    config.enforcement_enabled = enforcement_enabled
    config.daily_token_budget = daily_token_budget
    config.daily_query_budget = daily_query_budget
    return config


def _make_capability_enum(value: str) -> MagicMock:
    """Create a mock enum with .value attribute."""
    mock_enum = MagicMock()
    mock_enum.value = value
    return mock_enum


# ---------------------------------------------------------------------------
# Date range validation
# ---------------------------------------------------------------------------


class TestDateRangeValidation:
    """Tests for date range parameter validation."""

    async def test_validate_date_range_start_after_end_raises(self) -> None:
        """Rejects date ranges where start > end."""
        db = AsyncMock()
        service = UsageAggregationService(db)

        with pytest.raises(ValidationError, match="period_start must be <= period_end"):
            await service.get_summary(
                period_start=date(2025, 3, 15),
                period_end=date(2025, 3, 14),
            )

    async def test_validate_date_range_exceeds_max_days_raises(self) -> None:
        """Rejects date ranges exceeding 365 days."""
        db = AsyncMock()
        service = UsageAggregationService(db)

        with pytest.raises(ValidationError, match="must not exceed 365 days"):
            await service.get_summary(
                period_start=date(2024, 1, 1),
                period_end=date(2025, 3, 15),
            )

    async def test_validate_date_range_exactly_365_days_allowed(self) -> None:
        """Allows date ranges of exactly 365 days."""
        start = date(2025, 1, 1)
        end = start + timedelta(days=365)
        row = (1000, 500, 500, 10, 5, 3)
        db = _mock_db_returning_one(row)
        service = UsageAggregationService(db)

        result = await service.get_summary(period_start=start, period_end=end)
        assert result.total_tokens == 1000

    async def test_validate_date_range_same_day_allowed(self) -> None:
        """Allows start == end (single day range)."""
        today = date(2025, 3, 15)
        row = (100, 60, 40, 2, 1, 1)
        db = _mock_db_returning_one(row)
        service = UsageAggregationService(db)

        result = await service.get_summary(period_start=today, period_end=today)
        assert result.total_tokens == 100


# ---------------------------------------------------------------------------
# get_summary
# ---------------------------------------------------------------------------


class TestGetSummary:
    """Tests for the get_summary aggregation method."""

    async def test_get_summary_returns_correct_totals(self) -> None:
        """Returns correct aggregated totals from the database row."""
        row = (1500, 800, 700, 20, 5, 3)
        db = _mock_db_returning_one(row)
        service = UsageAggregationService(db)

        result = await service.get_summary(
            period_start=date(2025, 3, 1),
            period_end=date(2025, 3, 15),
        )

        assert isinstance(result, UsageSummaryResponse)
        assert result.total_tokens == 1500
        assert result.input_tokens == 800
        assert result.output_tokens == 700
        assert result.total_queries == 20
        assert result.total_documents_processed == 5
        assert result.unique_users == 3
        assert result.period_start == date(2025, 3, 1)
        assert result.period_end == date(2025, 3, 15)

    async def test_get_summary_empty_period_returns_zeros(self) -> None:
        """Returns zeros when no data exists for the period."""
        row = (0, 0, 0, 0, 0, 0)
        db = _mock_db_returning_one(row)
        service = UsageAggregationService(db)

        result = await service.get_summary(
            period_start=date(2025, 3, 1),
            period_end=date(2025, 3, 15),
        )

        assert result.total_tokens == 0
        assert result.total_queries == 0
        assert result.unique_users == 0


# ---------------------------------------------------------------------------
# get_trends
# ---------------------------------------------------------------------------


class TestGetTrends:
    """Tests for the get_trends daily aggregation method."""

    async def test_get_trends_returns_daily_rows(self) -> None:
        """Returns one DailyUsageResponse per day with data."""
        rows = [
            (date(2025, 3, 1), 500, 10, 2),
            (date(2025, 3, 2), 700, 15, 3),
        ]
        db = _mock_db_returning_rows(rows)
        service = UsageAggregationService(db)

        result = await service.get_trends(
            period_start=date(2025, 3, 1),
            period_end=date(2025, 3, 2),
        )

        assert len(result) == 2
        assert isinstance(result[0], DailyUsageResponse)
        assert result[0].date == date(2025, 3, 1)
        assert result[0].total_tokens == 500
        assert result[0].query_count == 10
        assert result[0].document_count == 2
        assert result[1].total_tokens == 700

    async def test_get_trends_empty_period_returns_empty(self) -> None:
        """Returns empty list when no data exists for the period."""
        db = _mock_db_returning_rows([])
        service = UsageAggregationService(db)

        result = await service.get_trends(
            period_start=date(2025, 3, 1),
            period_end=date(2025, 3, 2),
        )

        assert result == []


# ---------------------------------------------------------------------------
# get_by_capability
# ---------------------------------------------------------------------------


class TestGetByCapability:
    """Tests for the get_by_capability breakdown method."""

    async def test_get_by_capability_returns_breakdown(self) -> None:
        """Returns one CapabilityUsageResponse per capability with data."""
        rows = [
            (_make_capability_enum("answer_synthesis"), 1000, 20, 600, 400),
            (_make_capability_enum("intent_analysis"), 300, 15, 200, 100),
        ]
        db = _mock_db_returning_rows(rows)
        service = UsageAggregationService(db)

        result = await service.get_by_capability(
            period_start=date(2025, 3, 1),
            period_end=date(2025, 3, 15),
        )

        assert len(result) == 2
        assert isinstance(result[0], CapabilityUsageResponse)
        assert result[0].capability == "answer_synthesis"
        assert result[0].total_tokens == 1000
        assert result[0].call_count == 20
        assert result[0].input_tokens == 600
        assert result[0].output_tokens == 400

    async def test_get_by_capability_empty_returns_empty(self) -> None:
        """Returns empty list when no usage in the period."""
        db = _mock_db_returning_rows([])
        service = UsageAggregationService(db)

        result = await service.get_by_capability(
            period_start=date(2025, 3, 1),
            period_end=date(2025, 3, 15),
        )

        assert result == []


# ---------------------------------------------------------------------------
# get_by_game
# ---------------------------------------------------------------------------


class TestGetByGame:
    """Tests for the get_by_game breakdown method."""

    async def test_get_by_game_returns_breakdown(self) -> None:
        """Returns one GameUsageResponse per game with data."""
        game_id = uuid.uuid4()
        rows = [
            (game_id, "Catan", 800, 200, 15),
        ]
        db = _mock_db_returning_rows(rows)
        service = UsageAggregationService(db)

        result = await service.get_by_game(
            period_start=date(2025, 3, 1),
            period_end=date(2025, 3, 15),
        )

        assert len(result) == 1
        assert isinstance(result[0], GameUsageResponse)
        assert result[0].game_id == str(game_id)
        assert result[0].game_name == "Catan"
        assert result[0].query_tokens == 800
        assert result[0].ingestion_tokens == 200
        assert result[0].query_count == 15


# ---------------------------------------------------------------------------
# get_by_user
# ---------------------------------------------------------------------------


class TestGetByUser:
    """Tests for the get_by_user breakdown method."""

    async def test_get_by_user_returns_breakdown(self) -> None:
        """Returns one UserUsageResponse per user with data."""
        user_id = uuid.uuid4()
        rows = [
            (user_id, "Alice", 1200, 25),
        ]
        db = _mock_db_returning_rows(rows)
        service = UsageAggregationService(db)

        result = await service.get_by_user(
            period_start=date(2025, 3, 1),
            period_end=date(2025, 3, 15),
        )

        assert len(result) == 1
        assert isinstance(result[0], UserUsageResponse)
        assert result[0].user_id == str(user_id)
        assert result[0].display_name == "Alice"
        assert result[0].total_tokens == 1200
        assert result[0].query_count == 25


# ---------------------------------------------------------------------------
# get_guardrail_status
# ---------------------------------------------------------------------------


class TestGetGuardrailStatus:
    """Tests for guardrail status computation and thresholds."""

    async def test_get_guardrail_status_no_config_returns_disabled(self) -> None:
        """Returns disabled status when no guardrail config exists."""
        db = _mock_db_for_guardrail(config=None)
        service = UsageAggregationService(db)

        result = await service.get_guardrail_status()

        assert isinstance(result, GuardrailStatusResponse)
        assert result.enforcement_enabled is False
        assert result.indicators == []

    async def test_get_guardrail_status_enforcement_disabled(self) -> None:
        """Indicators show disabled when enforcement is off."""
        config = _make_guardrail_config(
            enforcement_enabled=False,
            daily_token_budget=10000,
            daily_query_budget=100,
        )
        db = _mock_db_for_guardrail(config=config, daily_tokens=5000, daily_queries=50)
        service = UsageAggregationService(db)

        result = await service.get_guardrail_status()

        assert result.enforcement_enabled is False
        for indicator in result.indicators:
            assert indicator.status == GuardrailThreshold.DISABLED

    async def test_get_guardrail_status_green_threshold(self) -> None:
        """Indicators show green when usage < 70% of limit."""
        config = _make_guardrail_config(
            enforcement_enabled=True,
            daily_token_budget=10000,
            daily_query_budget=100,
        )
        # 50% usage -> green
        db = _mock_db_for_guardrail(config=config, daily_tokens=5000, daily_queries=50)
        service = UsageAggregationService(db)

        result = await service.get_guardrail_status()

        assert result.enforcement_enabled is True
        token_indicator = result.indicators[0]
        assert token_indicator.name == "Daily Token Budget"
        assert token_indicator.current == 5000
        assert token_indicator.limit == 10000
        assert token_indicator.status == GuardrailThreshold.GREEN

        query_indicator = result.indicators[1]
        assert query_indicator.status == GuardrailThreshold.GREEN

    async def test_get_guardrail_status_amber_threshold(self) -> None:
        """Indicators show amber when usage is 70-90% of limit."""
        config = _make_guardrail_config(
            enforcement_enabled=True,
            daily_token_budget=10000,
            daily_query_budget=100,
        )
        # 80% usage -> amber
        db = _mock_db_for_guardrail(config=config, daily_tokens=8000, daily_queries=80)
        service = UsageAggregationService(db)

        result = await service.get_guardrail_status()

        assert result.indicators[0].status == GuardrailThreshold.AMBER
        assert result.indicators[1].status == GuardrailThreshold.AMBER

    async def test_get_guardrail_status_red_threshold(self) -> None:
        """Indicators show red when usage >= 90% of limit."""
        config = _make_guardrail_config(
            enforcement_enabled=True,
            daily_token_budget=10000,
            daily_query_budget=100,
        )
        # 95% usage -> red
        db = _mock_db_for_guardrail(config=config, daily_tokens=9500, daily_queries=95)
        service = UsageAggregationService(db)

        result = await service.get_guardrail_status()

        assert result.indicators[0].status == GuardrailThreshold.RED
        assert result.indicators[1].status == GuardrailThreshold.RED

    async def test_get_guardrail_status_null_limits_disabled(self) -> None:
        """Indicators show disabled when limit is NULL."""
        config = _make_guardrail_config(
            enforcement_enabled=True,
            daily_token_budget=None,
            daily_query_budget=None,
        )
        db = _mock_db_for_guardrail(config=config, daily_tokens=5000, daily_queries=50)
        service = UsageAggregationService(db)

        result = await service.get_guardrail_status()

        for indicator in result.indicators:
            assert indicator.status == GuardrailThreshold.DISABLED

    async def test_get_guardrail_status_exactly_at_70_percent_is_amber(self) -> None:
        """70% of limit is the amber boundary (inclusive)."""
        config = _make_guardrail_config(
            enforcement_enabled=True,
            daily_token_budget=1000,
        )
        db = _mock_db_for_guardrail(config=config, daily_tokens=700, daily_queries=0)
        service = UsageAggregationService(db)

        result = await service.get_guardrail_status()

        assert result.indicators[0].status == GuardrailThreshold.AMBER

    async def test_get_guardrail_status_exactly_at_90_percent_is_red(self) -> None:
        """90% of limit is the red boundary (inclusive)."""
        config = _make_guardrail_config(
            enforcement_enabled=True,
            daily_token_budget=1000,
        )
        db = _mock_db_for_guardrail(config=config, daily_tokens=900, daily_queries=0)
        service = UsageAggregationService(db)

        result = await service.get_guardrail_status()

        assert result.indicators[0].status == GuardrailThreshold.RED

    async def test_get_guardrail_status_zero_limit_is_red(self) -> None:
        """A limit of 0 always shows red (division by zero protection)."""
        config = _make_guardrail_config(
            enforcement_enabled=True,
            daily_token_budget=0,
        )
        db = _mock_db_for_guardrail(config=config, daily_tokens=0, daily_queries=0)
        service = UsageAggregationService(db)

        result = await service.get_guardrail_status()

        assert result.indicators[0].status == GuardrailThreshold.RED


# ---------------------------------------------------------------------------
# Threshold computation (static method)
# ---------------------------------------------------------------------------


class TestComputeThreshold:
    """Tests for the static _compute_threshold method."""

    def test_compute_threshold_disabled_when_no_limit(self) -> None:
        """Returns DISABLED when limit is None."""
        result = UsageAggregationService._compute_threshold(
            current=100, limit=None, enforcement_enabled=True
        )
        assert result == GuardrailThreshold.DISABLED

    def test_compute_threshold_disabled_when_enforcement_off(self) -> None:
        """Returns DISABLED when enforcement is disabled."""
        result = UsageAggregationService._compute_threshold(
            current=900, limit=1000, enforcement_enabled=False
        )
        assert result == GuardrailThreshold.DISABLED

    def test_compute_threshold_green_below_70(self) -> None:
        """Returns GREEN when usage is below 70% of limit."""
        result = UsageAggregationService._compute_threshold(
            current=69, limit=100, enforcement_enabled=True
        )
        assert result == GuardrailThreshold.GREEN

    def test_compute_threshold_amber_at_70(self) -> None:
        """Returns AMBER at exactly 70% of limit."""
        result = UsageAggregationService._compute_threshold(
            current=70, limit=100, enforcement_enabled=True
        )
        assert result == GuardrailThreshold.AMBER

    def test_compute_threshold_amber_between_70_and_90(self) -> None:
        """Returns AMBER between 70% and 90% of limit."""
        result = UsageAggregationService._compute_threshold(
            current=85, limit=100, enforcement_enabled=True
        )
        assert result == GuardrailThreshold.AMBER

    def test_compute_threshold_red_at_90(self) -> None:
        """Returns RED at exactly 90% of limit."""
        result = UsageAggregationService._compute_threshold(
            current=90, limit=100, enforcement_enabled=True
        )
        assert result == GuardrailThreshold.RED

    def test_compute_threshold_red_above_100(self) -> None:
        """Returns RED when usage exceeds limit."""
        result = UsageAggregationService._compute_threshold(
            current=150, limit=100, enforcement_enabled=True
        )
        assert result == GuardrailThreshold.RED

    def test_compute_threshold_zero_limit_is_red(self) -> None:
        """Returns RED when limit is zero."""
        result = UsageAggregationService._compute_threshold(
            current=0, limit=0, enforcement_enabled=True
        )
        assert result == GuardrailThreshold.RED

    def test_compute_threshold_zero_usage_is_green(self) -> None:
        """Returns GREEN when usage is zero with positive limit."""
        result = UsageAggregationService._compute_threshold(
            current=0, limit=1000, enforcement_enabled=True
        )
        assert result == GuardrailThreshold.GREEN


# ---------------------------------------------------------------------------
# Date range timestamp conversion
# ---------------------------------------------------------------------------


class TestDateRangeTimestamps:
    """Tests for the _date_range_timestamps helper."""

    def test_single_day_range(self) -> None:
        """Single day produces start of day to start of next day."""
        start, end = UsageAggregationService._date_range_timestamps(
            date(2025, 3, 15), date(2025, 3, 15)
        )
        assert start == "2025-03-15"
        assert end == "2025-03-16"

    def test_multi_day_range(self) -> None:
        """Multi-day range has exclusive end."""
        start, end = UsageAggregationService._date_range_timestamps(
            date(2025, 3, 1), date(2025, 3, 15)
        )
        assert start == "2025-03-01"
        assert end == "2025-03-16"


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestSchemas:
    """Tests for Pydantic response schema validation."""

    def test_usage_summary_rejects_negative_tokens(self) -> None:
        """UsageSummaryResponse rejects negative token counts."""
        with pytest.raises(ValueError):
            UsageSummaryResponse(
                total_tokens=-1,
                total_queries=0,
                total_documents_processed=0,
                input_tokens=0,
                output_tokens=0,
                unique_users=0,
                period_start=date(2025, 3, 1),
                period_end=date(2025, 3, 15),
            )

    def test_guardrail_threshold_enum_values(self) -> None:
        """GuardrailThreshold enum has all expected values."""
        assert set(GuardrailThreshold) == {
            GuardrailThreshold.GREEN,
            GuardrailThreshold.AMBER,
            GuardrailThreshold.RED,
            GuardrailThreshold.DISABLED,
        }
