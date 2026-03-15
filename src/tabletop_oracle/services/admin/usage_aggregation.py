"""Usage aggregation service for the admin dashboard.

Aggregates token usage data from ``token_usage_log`` with joins to
``sessions``, ``games``, and ``users`` for contextual breakdowns.
Reads ``guardrail_config`` for threshold status computation.

All methods are read-only aggregation queries. No data is written.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import case, cast, func, select
from sqlalchemy.types import Date

from tabletop_oracle.errors.exceptions import ValidationError
from tabletop_oracle.models.game import Game
from tabletop_oracle.models.guardrail import GuardrailConfig
from tabletop_oracle.models.model_slot import TokenUsageLog
from tabletop_oracle.models.session import Session
from tabletop_oracle.models.user import User
from tabletop_oracle.schemas.usage import (
    CapabilityUsageResponse,
    DailyUsageResponse,
    GameUsageResponse,
    GuardrailIndicator,
    GuardrailStatusResponse,
    GuardrailThreshold,
    UsageSummaryResponse,
    UserUsageResponse,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

_MAX_RANGE_DAYS = 365


class UsageAggregationService:
    """Aggregates token usage data for the admin dashboard.

    Reads from ``token_usage_log`` and joins to ``sessions`` (for game_id)
    and ``users`` (for display names). All methods accept a date range
    for filtering.

    Args:
        db: SQLAlchemy async session for database operations.
    """

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get_summary(
        self,
        period_start: date,
        period_end: date,
    ) -> UsageSummaryResponse:
        """Get current period summary.

        Args:
            period_start: Inclusive start of the aggregation period.
            period_end: Inclusive end of the aggregation period.

        Returns:
            UsageSummaryResponse with total tokens, queries, documents,
            input/output breakdown, and unique user count.

        Raises:
            ValidationError: If the date range is invalid.
        """
        self._validate_date_range(period_start, period_end)

        start_ts, end_ts = self._date_range_timestamps(period_start, period_end)

        stmt = select(
            func.coalesce(func.sum(TokenUsageLog.input_tokens + TokenUsageLog.output_tokens), 0),
            func.coalesce(func.sum(TokenUsageLog.input_tokens), 0),
            func.coalesce(func.sum(TokenUsageLog.output_tokens), 0),
            func.count().filter(TokenUsageLog.session_id.isnot(None)),
            func.count().filter(TokenUsageLog.document_id.isnot(None)),
            func.count(func.distinct(TokenUsageLog.user_id)),
        ).where(
            TokenUsageLog.created_at >= start_ts,
            TokenUsageLog.created_at < end_ts,
        )
        result = await self._db.execute(stmt)
        row = result.one()

        return UsageSummaryResponse(
            total_tokens=int(row[0]),
            input_tokens=int(row[1]),
            output_tokens=int(row[2]),
            total_queries=int(row[3]),
            total_documents_processed=int(row[4]),
            unique_users=int(row[5]),
            period_start=period_start,
            period_end=period_end,
        )

    async def get_trends(
        self,
        period_start: date,
        period_end: date,
    ) -> list[DailyUsageResponse]:
        """Get daily usage data for trend chart.

        Args:
            period_start: Inclusive start of the aggregation period.
            period_end: Inclusive end of the aggregation period.

        Returns:
            List of DailyUsageResponse ordered by date ascending.

        Raises:
            ValidationError: If the date range is invalid.
        """
        self._validate_date_range(period_start, period_end)

        start_ts, end_ts = self._date_range_timestamps(period_start, period_end)
        day_col = cast(TokenUsageLog.created_at, Date).label("day")

        stmt = (
            select(
                day_col,
                func.coalesce(
                    func.sum(TokenUsageLog.input_tokens + TokenUsageLog.output_tokens), 0
                ),
                func.count().filter(TokenUsageLog.session_id.isnot(None)),
                func.count().filter(TokenUsageLog.document_id.isnot(None)),
            )
            .where(
                TokenUsageLog.created_at >= start_ts,
                TokenUsageLog.created_at < end_ts,
            )
            .group_by(day_col)
            .order_by(day_col)
        )
        result = await self._db.execute(stmt)

        return [
            DailyUsageResponse(
                date=row[0],
                total_tokens=int(row[1]),
                query_count=int(row[2]),
                document_count=int(row[3]),
            )
            for row in result.all()
        ]

    async def get_by_capability(
        self,
        period_start: date,
        period_end: date,
    ) -> list[CapabilityUsageResponse]:
        """Get usage breakdown by model capability.

        Args:
            period_start: Inclusive start of the aggregation period.
            period_end: Inclusive end of the aggregation period.

        Returns:
            List of CapabilityUsageResponse grouped by capability.

        Raises:
            ValidationError: If the date range is invalid.
        """
        self._validate_date_range(period_start, period_end)

        start_ts, end_ts = self._date_range_timestamps(period_start, period_end)

        stmt = (
            select(
                TokenUsageLog.capability,
                func.coalesce(
                    func.sum(TokenUsageLog.input_tokens + TokenUsageLog.output_tokens), 0
                ),
                func.count(),
                func.coalesce(func.sum(TokenUsageLog.input_tokens), 0),
                func.coalesce(func.sum(TokenUsageLog.output_tokens), 0),
            )
            .where(
                TokenUsageLog.created_at >= start_ts,
                TokenUsageLog.created_at < end_ts,
            )
            .group_by(TokenUsageLog.capability)
            .order_by(TokenUsageLog.capability)
        )
        result = await self._db.execute(stmt)

        return [
            CapabilityUsageResponse(
                capability=row[0].value,
                total_tokens=int(row[1]),
                call_count=int(row[2]),
                input_tokens=int(row[3]),
                output_tokens=int(row[4]),
            )
            for row in result.all()
        ]

    async def get_by_game(
        self,
        period_start: date,
        period_end: date,
    ) -> list[GameUsageResponse]:
        """Get usage breakdown by game.

        Joins ``token_usage_log`` -> ``sessions`` -> ``games`` for game names.
        Only includes entries that have a session_id (query usage) or
        document_id (ingestion usage linked via session context).

        Args:
            period_start: Inclusive start of the aggregation period.
            period_end: Inclusive end of the aggregation period.

        Returns:
            List of GameUsageResponse grouped by game.

        Raises:
            ValidationError: If the date range is invalid.
        """
        self._validate_date_range(period_start, period_end)

        start_ts, end_ts = self._date_range_timestamps(period_start, period_end)

        # Query usage: entries with session_id, joined to sessions for game_id
        stmt = (
            select(
                Game.id,
                Game.name,
                func.coalesce(
                    func.sum(
                        case(
                            (
                                TokenUsageLog.session_id.isnot(None),
                                TokenUsageLog.input_tokens + TokenUsageLog.output_tokens,
                            ),
                            else_=0,
                        )
                    ),
                    0,
                ),
                func.coalesce(
                    func.sum(
                        case(
                            (
                                TokenUsageLog.document_id.isnot(None),
                                TokenUsageLog.input_tokens + TokenUsageLog.output_tokens,
                            ),
                            else_=0,
                        )
                    ),
                    0,
                ),
                func.count().filter(TokenUsageLog.session_id.isnot(None)),
            )
            .join(Session, TokenUsageLog.session_id == Session.id)
            .join(Game, Session.game_id == Game.id)
            .where(
                TokenUsageLog.created_at >= start_ts,
                TokenUsageLog.created_at < end_ts,
            )
            .group_by(Game.id, Game.name)
            .order_by(Game.name)
        )
        result = await self._db.execute(stmt)

        return [
            GameUsageResponse(
                game_id=str(row[0]),
                game_name=row[1],
                query_tokens=int(row[2]),
                ingestion_tokens=int(row[3]),
                query_count=int(row[4]),
            )
            for row in result.all()
        ]

    async def get_by_user(
        self,
        period_start: date,
        period_end: date,
    ) -> list[UserUsageResponse]:
        """Get usage breakdown by user.

        Args:
            period_start: Inclusive start of the aggregation period.
            period_end: Inclusive end of the aggregation period.

        Returns:
            List of UserUsageResponse grouped by user.

        Raises:
            ValidationError: If the date range is invalid.
        """
        self._validate_date_range(period_start, period_end)

        start_ts, end_ts = self._date_range_timestamps(period_start, period_end)

        stmt = (
            select(
                User.id,
                User.display_name,
                func.coalesce(
                    func.sum(TokenUsageLog.input_tokens + TokenUsageLog.output_tokens), 0
                ),
                func.count(),
            )
            .join(User, TokenUsageLog.user_id == User.id)
            .where(
                TokenUsageLog.created_at >= start_ts,
                TokenUsageLog.created_at < end_ts,
            )
            .group_by(User.id, User.display_name)
            .order_by(User.display_name)
        )
        result = await self._db.execute(stmt)

        return [
            UserUsageResponse(
                user_id=str(row[0]),
                display_name=row[1],
                total_tokens=int(row[2]),
                query_count=int(row[3]),
            )
            for row in result.all()
        ]

    async def get_guardrail_status(self) -> GuardrailStatusResponse:
        """Get current guardrail status with usage vs limits.

        Reads ``guardrail_config`` for limits and today's token/query
        totals. Computes traffic-light status per the design thresholds:
        green < 70%, amber 70-90%, red >= 90%.

        Returns:
            GuardrailStatusResponse with per-guardrail indicators.
        """
        config = await self._load_guardrail_config()

        if config is None:
            return GuardrailStatusResponse(
                enforcement_enabled=False,
                indicators=[],
            )

        # Get today's usage totals
        daily_tokens = await self._get_daily_token_total()
        daily_queries = await self._get_daily_query_count()

        indicators = [
            GuardrailIndicator(
                name="Daily Token Budget",
                current=daily_tokens,
                limit=config.daily_token_budget,
                status=self._compute_threshold(
                    current=daily_tokens,
                    limit=config.daily_token_budget,
                    enforcement_enabled=config.enforcement_enabled,
                ),
            ),
            GuardrailIndicator(
                name="Daily Query Budget",
                current=daily_queries,
                limit=config.daily_query_budget,
                status=self._compute_threshold(
                    current=daily_queries,
                    limit=config.daily_query_budget,
                    enforcement_enabled=config.enforcement_enabled,
                ),
            ),
        ]

        return GuardrailStatusResponse(
            enforcement_enabled=config.enforcement_enabled,
            indicators=indicators,
        )

    @staticmethod
    def _validate_date_range(period_start: date, period_end: date) -> None:
        """Validate the date range parameters.

        Args:
            period_start: Start of the period (inclusive).
            period_end: End of the period (inclusive).

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

    @staticmethod
    def _date_range_timestamps(period_start: date, period_end: date) -> tuple[str, str]:
        """Convert inclusive date range to timestamp boundaries.

        The start is midnight of period_start, the end is midnight of
        the day after period_end (exclusive upper bound).

        Args:
            period_start: Inclusive start date.
            period_end: Inclusive end date.

        Returns:
            Tuple of (start_iso, end_iso) suitable for timestamp comparison.
        """
        start = period_start.isoformat()
        end = (period_end + timedelta(days=1)).isoformat()
        return start, end

    @staticmethod
    def _compute_threshold(
        *,
        current: int,
        limit: int | None,
        enforcement_enabled: bool,
    ) -> GuardrailThreshold:
        """Compute the traffic-light threshold for a guardrail metric.

        Args:
            current: Current usage value.
            limit: Configured limit, or None if not set.
            enforcement_enabled: Master enforcement toggle.

        Returns:
            GuardrailThreshold enum value.
        """
        if limit is None or not enforcement_enabled:
            return GuardrailThreshold.DISABLED

        if limit == 0:
            return GuardrailThreshold.RED

        ratio = current / limit
        if ratio >= 0.9:
            return GuardrailThreshold.RED
        if ratio >= 0.7:
            return GuardrailThreshold.AMBER
        return GuardrailThreshold.GREEN

    async def _load_guardrail_config(self) -> GuardrailConfig | None:
        """Load the single-row guardrail configuration.

        Returns:
            GuardrailConfig or None if no configuration exists.
        """
        stmt = select(GuardrailConfig).limit(1)
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def _get_daily_token_total(self) -> int:
        """Get total tokens consumed today (UTC).

        Returns:
            Total token count for the current day.
        """
        today = date.today()
        start, end = self._date_range_timestamps(today, today)
        stmt = select(
            func.coalesce(func.sum(TokenUsageLog.input_tokens + TokenUsageLog.output_tokens), 0)
        ).where(
            TokenUsageLog.created_at >= start,
            TokenUsageLog.created_at < end,
        )
        result = await self._db.execute(stmt)
        return int(result.scalar_one())

    async def _get_daily_query_count(self) -> int:
        """Get number of query-attributed entries today (UTC).

        Counts token_usage_log entries with a session_id (indicating
        they came from a user query rather than document ingestion).

        Returns:
            Number of query entries today.
        """
        today = date.today()
        start, end = self._date_range_timestamps(today, today)
        stmt = select(func.count()).where(
            TokenUsageLog.created_at >= start,
            TokenUsageLog.created_at < end,
            TokenUsageLog.session_id.isnot(None),
        )
        result = await self._db.execute(stmt)
        return int(result.scalar_one())
