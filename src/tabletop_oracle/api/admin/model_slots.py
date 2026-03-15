"""Model slot admin API — list and update AI model configurations.

All endpoints require curator role. Mounted under
``/api/v1/admin/model-slots`` via the main router.

Endpoints:
    GET  /              — list all model slot configurations
    PUT  /{capability}  — partially update a single slot
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request

from tabletop_oracle.api.deps import DbSession  # noqa: TC001
from tabletop_oracle.api.response import ok
from tabletop_oracle.auth.dependencies import require_role
from tabletop_oracle.errors.exceptions import NotFoundError
from tabletop_oracle.models.enums import ModelCapability
from tabletop_oracle.schemas.common import DataEnvelope  # noqa: TC001
from tabletop_oracle.schemas.model_slot import ModelSlotResponse, ModelSlotUpdate
from tabletop_oracle.services.model.slot_repository import ModelSlotRepository
from tabletop_oracle.services.model.slot_service import ModelSlotService

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["admin"],
    dependencies=[Depends(require_role("curator"))],
)


def _get_slot_service(db: DbSession) -> ModelSlotService:
    """Build a ModelSlotService wired to the request's database session.

    Args:
        db: Async database session from the request-scoped dependency.

    Returns:
        ModelSlotService instance backed by a ModelSlotRepository.
    """
    return ModelSlotService(ModelSlotRepository(db))


def _capability_from_path(capability: str) -> ModelCapability:
    """Parse a capability path parameter into a ModelCapability enum.

    Args:
        capability: The raw path parameter string.

    Returns:
        The corresponding ModelCapability enum value.

    Raises:
        NotFoundError: If the string does not match any capability.
    """
    try:
        return ModelCapability(capability)
    except ValueError as err:
        raise NotFoundError("Model slot", capability) from err


@router.get("")
async def list_model_slots(
    request: Request,
    db: DbSession,
) -> DataEnvelope[object]:
    """List all model slot configurations.

    Returns all capability slots with their current configuration.
    Requires curator role.

    Args:
        request: Incoming HTTP request (provides request_id).
        db: Database session.

    Returns:
        DataEnvelope containing list of ModelSlotResponse items.
    """
    service = _get_slot_service(db)
    configs = await service.list_all()

    items = [
        ModelSlotResponse(
            capability=cfg.capability.value,
            provider=cfg.provider,
            model_id=cfg.model_id,
            max_tokens_per_call=cfg.max_tokens_per_call,
            temperature=cfg.temperature,
            fallback_provider=cfg.fallback_provider,
            fallback_model_id=cfg.fallback_model_id,
        )
        for cfg in configs
    ]
    return ok(items, request)


@router.put("/{capability}")
async def update_model_slot(
    capability: str,
    body: ModelSlotUpdate,
    request: Request,
    db: DbSession,
) -> DataEnvelope[object]:
    """Update a model slot configuration (partial update).

    Only fields present in the request body are modified. The capability
    path parameter must match a valid ``model_capability`` enum value.
    Requires curator role.

    Args:
        capability: The model capability to update (path parameter).
        body: Partial update payload with UNSET-aware fields.
        request: Incoming HTTP request (provides request_id).
        db: Database session.

    Returns:
        DataEnvelope containing the updated ModelSlotResponse.

    Raises:
        NotFoundError: If the capability is not valid or not configured.
    """
    cap = _capability_from_path(capability)
    service = _get_slot_service(db)

    updates = body.to_update_dict()
    config = await service.update_slot(cap, updates)

    response = ModelSlotResponse(
        capability=config.capability.value,
        provider=config.provider,
        model_id=config.model_id,
        max_tokens_per_call=config.max_tokens_per_call,
        temperature=config.temperature,
        fallback_provider=config.fallback_provider,
        fallback_model_id=config.fallback_model_id,
    )
    return ok(response, request)
