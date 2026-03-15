"""Quality metrics admin API -- read-only quality signals for the dashboard.

All endpoints require curator role. Mounted under
``/api/v1/admin/quality`` via the main router.

Endpoints:
    GET /metrics  -- per-game quality metrics for a date range
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
from tabletop_oracle.services.admin.quality_metrics import QualityMetricsService

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["admin"],
    dependencies=[Depends(require_role("curator"))],
)

PeriodStart = Annotated[date, Query(description="Start of the period (inclusive).")]
PeriodEnd = Annotated[date, Query(description="End of the period (inclusive).")]


def _get_service(db: DbSession) -> QualityMetricsService:
    """Build a QualityMetricsService wired to the request's database session.

    Args:
        db: Async database session from the request-scoped dependency.

    Returns:
        QualityMetricsService instance.
    """
    return QualityMetricsService(db)


@router.get("/metrics")
async def get_quality_metrics(
    request: Request,
    db: DbSession,
    period_start: PeriodStart,
    period_end: PeriodEnd,
) -> DataEnvelope[object]:
    """Get quality metrics per game for a date range.

    Args:
        request: Incoming HTTP request (provides request_id).
        db: Database session.
        period_start: Inclusive start of the aggregation period.
        period_end: Inclusive end of the aggregation period.

    Returns:
        DataEnvelope containing list of GameQualityMetricsResponse.
    """
    validate_date_range(period_start, period_end)
    service = _get_service(db)
    metrics = await service.get_metrics(period_start, period_end)
    return ok(metrics, request)
