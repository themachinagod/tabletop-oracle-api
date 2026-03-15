"""Usage aggregation admin API -- read-only usage metrics for the dashboard.

All endpoints require curator role. Mounted under
``/api/v1/admin/usage`` via the main router.

Endpoints:
    GET /summary          -- period summary totals
    GET /trends           -- daily usage for trend charts
    GET /by-capability    -- breakdown by AI capability
    GET /by-game          -- breakdown by game
    GET /by-user          -- breakdown by user
    GET /guardrail-status -- current guardrail usage vs limits
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from tabletop_oracle.api.admin._date_range import validate_date_range
from tabletop_oracle.api.deps import DbSession  # noqa: TC001
from tabletop_oracle.api.response import ok
from tabletop_oracle.auth.dependencies import require_role
from tabletop_oracle.schemas.common import DataEnvelope  # noqa: TC001
from tabletop_oracle.services.admin.usage_aggregation import UsageAggregationService

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["admin"],
    dependencies=[Depends(require_role("curator"))],
)

PeriodStart = Annotated[date, Query(description="Start of the period (inclusive).")]
PeriodEnd = Annotated[date, Query(description="End of the period (inclusive).")]


def _get_service(db: DbSession) -> UsageAggregationService:
    """Build a UsageAggregationService wired to the request's database session.

    Args:
        db: Async database session from the request-scoped dependency.

    Returns:
        UsageAggregationService instance.
    """
    return UsageAggregationService(db)


@router.get("/summary")
async def get_usage_summary(
    request: Request,
    db: DbSession,
    period_start: PeriodStart,
    period_end: PeriodEnd,
) -> DataEnvelope[object]:
    """Get usage summary totals for a date range.

    Args:
        request: Incoming HTTP request (provides request_id).
        db: Database session.
        period_start: Inclusive start of the aggregation period.
        period_end: Inclusive end of the aggregation period.

    Returns:
        DataEnvelope containing UsageSummaryResponse.
    """
    validate_date_range(period_start, period_end)
    service = _get_service(db)
    summary = await service.get_summary(period_start, period_end)
    return ok(summary, request)


@router.get("/trends")
async def get_usage_trends(
    request: Request,
    db: DbSession,
    period_start: PeriodStart,
    period_end: PeriodEnd,
) -> DataEnvelope[object]:
    """Get daily usage data for trend charts.

    Args:
        request: Incoming HTTP request (provides request_id).
        db: Database session.
        period_start: Inclusive start of the aggregation period.
        period_end: Inclusive end of the aggregation period.

    Returns:
        DataEnvelope containing list of DailyUsageResponse.
    """
    validate_date_range(period_start, period_end)
    service = _get_service(db)
    trends = await service.get_trends(period_start, period_end)
    return ok(trends, request)


@router.get("/by-capability")
async def get_usage_by_capability(
    request: Request,
    db: DbSession,
    period_start: PeriodStart,
    period_end: PeriodEnd,
) -> DataEnvelope[object]:
    """Get usage breakdown by AI capability.

    Args:
        request: Incoming HTTP request (provides request_id).
        db: Database session.
        period_start: Inclusive start of the aggregation period.
        period_end: Inclusive end of the aggregation period.

    Returns:
        DataEnvelope containing list of CapabilityUsageResponse.
    """
    validate_date_range(period_start, period_end)
    service = _get_service(db)
    data = await service.get_by_capability(period_start, period_end)
    return ok(data, request)


@router.get("/by-game")
async def get_usage_by_game(
    request: Request,
    db: DbSession,
    period_start: PeriodStart,
    period_end: PeriodEnd,
) -> DataEnvelope[object]:
    """Get usage breakdown by game.

    Args:
        request: Incoming HTTP request (provides request_id).
        db: Database session.
        period_start: Inclusive start of the aggregation period.
        period_end: Inclusive end of the aggregation period.

    Returns:
        DataEnvelope containing list of GameUsageResponse.
    """
    validate_date_range(period_start, period_end)
    service = _get_service(db)
    data = await service.get_by_game(period_start, period_end)
    return ok(data, request)


@router.get("/by-user")
async def get_usage_by_user(
    request: Request,
    db: DbSession,
    period_start: PeriodStart,
    period_end: PeriodEnd,
) -> DataEnvelope[object]:
    """Get usage breakdown by user.

    Args:
        request: Incoming HTTP request (provides request_id).
        db: Database session.
        period_start: Inclusive start of the aggregation period.
        period_end: Inclusive end of the aggregation period.

    Returns:
        DataEnvelope containing list of UserUsageResponse.
    """
    validate_date_range(period_start, period_end)
    service = _get_service(db)
    data = await service.get_by_user(period_start, period_end)
    return ok(data, request)


@router.get("/guardrail-status")
async def get_guardrail_status(
    request: Request,
    db: DbSession,
) -> DataEnvelope[object]:
    """Get current guardrail usage vs configured limits.

    Args:
        request: Incoming HTTP request (provides request_id).
        db: Database session.

    Returns:
        DataEnvelope containing GuardrailStatusResponse.
    """
    service = _get_service(db)
    status = await service.get_guardrail_status()
    return ok(status, request)
