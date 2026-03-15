"""Guardrail config admin API -- read and update guardrail settings.

All endpoints require curator role. Mounted under
``/api/v1/admin/guardrail-config`` via the main router.

Endpoints:
    GET  /  -- get current guardrail configuration
    PUT  /  -- partially update guardrail configuration
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Request

from tabletop_oracle.api.deps import DbSession  # noqa: TC001
from tabletop_oracle.api.response import ok
from tabletop_oracle.auth.dependencies import require_role
from tabletop_oracle.schemas.common import DataEnvelope  # noqa: TC001
from tabletop_oracle.schemas.guardrail_config import (
    GuardrailConfigResponse,
    GuardrailConfigUpdate,
)
from tabletop_oracle.services.model.guardrail import GuardrailService
from tabletop_oracle.services.model.token_usage import TokenUsageService

if TYPE_CHECKING:
    from tabletop_oracle.models.guardrail import GuardrailConfig

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["admin"],
    dependencies=[Depends(require_role("curator"))],
)


def _get_guardrail_service(db: DbSession) -> GuardrailService:
    """Build a GuardrailService wired to the request's database session.

    The admin API only uses ``get_config()`` and ``update_config()``
    which do not require TokenUsageService. A stub instance is provided
    to satisfy the constructor signature.

    Args:
        db: Async database session from the request-scoped dependency.

    Returns:
        GuardrailService instance.
    """
    token_usage = TokenUsageService(db)
    return GuardrailService(db=db, token_usage_service=token_usage)


def _config_to_response(config: GuardrailConfig) -> GuardrailConfigResponse:
    """Convert a GuardrailConfig ORM entity to a response schema.

    Args:
        config: GuardrailConfig ORM entity.

    Returns:
        Populated GuardrailConfigResponse.
    """
    return GuardrailConfigResponse(
        enforcement_enabled=config.enforcement_enabled,
        max_tokens_per_query=config.max_tokens_per_query,
        max_model_calls_per_query=config.max_model_calls_per_query,
        max_queries_per_session=config.max_queries_per_session,
        daily_token_budget=config.daily_token_budget,
        daily_query_budget=config.daily_query_budget,
        per_document_ingestion_limit=config.per_document_ingestion_limit,
        updated_at=config.updated_at,
    )


@router.get("")
async def get_guardrail_config(
    request: Request,
    db: DbSession,
) -> DataEnvelope[object]:
    """Get current guardrail configuration.

    Returns the single-row guardrail configuration with all limit
    values and the enforcement toggle. Requires curator role.

    Args:
        request: Incoming HTTP request (provides request_id).
        db: Database session.

    Returns:
        DataEnvelope containing GuardrailConfigResponse.
    """
    service = _get_guardrail_service(db)
    config = await service.get_config()
    return ok(_config_to_response(config), request)


@router.put("")
async def update_guardrail_config(
    body: GuardrailConfigUpdate,
    request: Request,
    db: DbSession,
) -> DataEnvelope[object]:
    """Update guardrail configuration (partial update).

    Only fields present in the request body are modified. Requires
    curator role.

    Args:
        body: Partial update payload with UNSET-aware fields.
        request: Incoming HTTP request (provides request_id).
        db: Database session.

    Returns:
        DataEnvelope containing the updated GuardrailConfigResponse.
    """
    service = _get_guardrail_service(db)
    updates = body.to_update_dict()
    config = await service.update_config(updates)
    return ok(_config_to_response(config), request)
