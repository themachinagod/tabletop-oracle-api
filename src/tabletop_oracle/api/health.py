"""Health check endpoint with database connectivity verification.

Returns a structured response indicating service and dependency health.
The endpoint is unauthenticated — excluded from any auth middleware.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Request
from sqlalchemy import text
from starlette.responses import JSONResponse

from tabletop_oracle.api.deps import DbSession  # noqa: TC001
from tabletop_oracle.schemas.common import DataEnvelope, ResponseMeta

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


class HealthCheckResult:
    """Encapsulates the outcome of all health sub-checks.

    Attributes:
        checks: Mapping of check name to status string.
    """

    def __init__(self) -> None:
        self.checks: dict[str, str] = {}

    @property
    def is_healthy(self) -> bool:
        """Return True when every sub-check reports healthy."""
        return all(v == "healthy" for v in self.checks.values())

    @property
    def overall_status(self) -> str:
        """Return 'healthy' if all checks pass, 'unhealthy' otherwise."""
        return "healthy" if self.is_healthy else "unhealthy"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the response payload shape."""
        return {
            "status": self.overall_status,
            "checks": dict(self.checks),
        }


async def _check_database(session: AsyncSession) -> str:
    """Execute a trivial query to verify database connectivity.

    Args:
        session: Async database session.

    Returns:
        'healthy' if the query succeeds, 'unhealthy' otherwise.
    """
    try:
        await session.execute(text("SELECT 1"))
        return "healthy"
    except Exception:
        logger.warning("Health check: database connectivity failed", exc_info=True)
        return "unhealthy"


@router.get("/health")
async def health_check(request: Request, session: DbSession) -> JSONResponse:
    """Return service health status with dependency checks.

    Performs a database connectivity check and returns a structured
    DataEnvelope response. Returns HTTP 200 when all checks pass,
    HTTP 503 when any check fails.

    Args:
        request: The incoming HTTP request (provides request_id via state).
        session: Async database session injected by FastAPI.

    Returns:
        JSONResponse with DataEnvelope-shaped body.
    """
    result = HealthCheckResult()
    result.checks["database"] = await _check_database(session)

    request_id: str = getattr(request.state, "request_id", "unknown")
    envelope = DataEnvelope[dict[str, Any]](
        data=result.to_dict(),
        meta=ResponseMeta(request_id=request_id),
    )

    status_code = 200 if result.is_healthy else 503
    return JSONResponse(content=envelope.model_dump(), status_code=status_code)
