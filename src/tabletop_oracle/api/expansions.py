"""Expansion API endpoints — CRUD, archive/restore under game silo.

All endpoints live under ``/api/v1/games/{game_id}/expansions`` (prefix
applied in router.py). Read endpoints require authentication; write
endpoints require curator role. Game silo isolation is enforced — every
expansion must belong to the specified game_id.

Responses follow the F001 envelope convention.
"""

from __future__ import annotations

import logging
import uuid  # noqa: TC003 — FastAPI needs UUID at runtime for path validation
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request, Response

from tabletop_oracle.api.deps import DbSession  # noqa: TC001
from tabletop_oracle.api.pagination import PaginationParams, get_pagination_params
from tabletop_oracle.api.response import ok, ok_list
from tabletop_oracle.auth.dependencies import CurrentUserDep, require_role
from tabletop_oracle.repositories.expansion_repository import ExpansionRepository
from tabletop_oracle.repositories.game_repository import GameRepository
from tabletop_oracle.schemas.common import DataEnvelope, ListEnvelope  # noqa: TC001
from tabletop_oracle.schemas.expansion import (
    ExpansionCreate,
    ExpansionFilters,
    ExpansionListResponse,
    ExpansionResponse,
    ExpansionUpdate,
)
from tabletop_oracle.services.expansion_service import ExpansionService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["expansions"])

#: Annotated dependency for pagination query parameters.
PaginationDep = Annotated[PaginationParams, Depends(get_pagination_params)]

#: Dependency that enforces curator role on write endpoints.
_curator = Depends(require_role("curator"))


def _get_expansion_service(db: DbSession) -> ExpansionService:
    """Build an ExpansionService wired to the request's database session.

    Args:
        db: Async database session from the request-scoped dependency.

    Returns:
        ExpansionService instance backed by repository dependencies.
    """
    return ExpansionService(
        expansion_repo=ExpansionRepository(db),
        game_repo=GameRepository(db),
    )


def _expansion_to_response(expansion: Any) -> ExpansionResponse:
    """Convert an Expansion ORM entity to an ExpansionResponse schema.

    Args:
        expansion: Expansion ORM entity.

    Returns:
        Populated ExpansionResponse schema.
    """
    return ExpansionResponse(
        id=expansion.id,
        game_id=expansion.game_id,
        name=expansion.name,
        year_published=expansion.year_published,
        description=expansion.description,
        archived_at=expansion.archived_at,
        created_at=expansion.created_at,
        updated_at=expansion.updated_at,
    )


@router.post(
    "",
    status_code=201,
    dependencies=[_curator],
)
async def create_expansion(
    game_id: uuid.UUID,
    body: ExpansionCreate,
    request: Request,
    response: Response,
    db: DbSession,
    user: CurrentUserDep,
) -> DataEnvelope[object]:
    """Create a new expansion under a game.

    Requires curator role. Validates that the parent game exists.
    Returns the created expansion in the standard envelope with a
    Location header.

    Args:
        game_id: UUID of the parent game.
        body: Validated expansion creation payload.
        request: Incoming HTTP request (provides request_id).
        response: Outgoing HTTP response (for Location header).
        db: Database session.
        user: Authenticated curator.

    Returns:
        DataEnvelope containing the created ExpansionResponse.

    Raises:
        NotFoundError: If the parent game does not exist.
    """
    service = _get_expansion_service(db)
    expansion = await service.create_expansion(game_id, body, user.user_id)

    expansion_response = _expansion_to_response(expansion)
    response.headers["Location"] = f"/api/v1/games/{game_id}/expansions/{expansion.id}"
    return ok(expansion_response, request)


@router.get("")
async def list_expansions(
    game_id: uuid.UUID,
    request: Request,
    db: DbSession,
    _user: CurrentUserDep,
    pagination: PaginationDep,
    archived: bool = Query(False, description="Include archived expansions"),
) -> ListEnvelope[object]:
    """List expansions for a game with filtering and pagination.

    Returns paginated results with document_count per expansion.
    By default, archived expansions are excluded.

    Args:
        game_id: UUID of the parent game.
        request: Incoming HTTP request (provides request_id).
        db: Database session.
        _user: Authenticated user (enforces auth, value unused).
        pagination: Parsed pagination parameters.
        archived: Whether to include archived expansions.

    Returns:
        ListEnvelope containing paginated ExpansionListResponse items.

    Raises:
        NotFoundError: If the parent game does not exist.
    """
    filters = ExpansionFilters(archived=archived)
    service = _get_expansion_service(db)
    summaries, total_count = await service.list_expansions(game_id, filters, pagination)

    items: list[object] = [
        ExpansionListResponse(
            id=s["expansion"].id,
            game_id=s["expansion"].game_id,
            name=s["expansion"].name,
            year_published=s["expansion"].year_published,
            description=s["expansion"].description,
            archived_at=s["expansion"].archived_at,
            created_at=s["expansion"].created_at,
            updated_at=s["expansion"].updated_at,
            document_count=s["document_count"],
        )
        for s in summaries
    ]

    return ok_list(items, total_count, pagination, request)


@router.get("/{expansion_id}")
async def get_expansion(
    game_id: uuid.UUID,
    expansion_id: uuid.UUID,
    request: Request,
    db: DbSession,
    _user: CurrentUserDep,
) -> DataEnvelope[object]:
    """Get expansion detail.

    Returns the full expansion resource. Enforces game silo isolation —
    returns 404 if the expansion does not belong to the specified game.

    Args:
        game_id: UUID of the parent game.
        expansion_id: UUID of the expansion to retrieve.
        request: Incoming HTTP request (provides request_id).
        db: Database session.
        _user: Authenticated user (enforces auth, value unused).

    Returns:
        DataEnvelope containing ExpansionResponse.

    Raises:
        NotFoundError: If the expansion does not exist or does not
            belong to the specified game.
    """
    service = _get_expansion_service(db)
    expansion = await service.get_expansion(game_id, expansion_id)
    return ok(_expansion_to_response(expansion), request)


@router.patch(
    "/{expansion_id}",
    dependencies=[_curator],
)
async def update_expansion(
    game_id: uuid.UUID,
    expansion_id: uuid.UUID,
    body: ExpansionUpdate,
    request: Request,
    db: DbSession,
) -> DataEnvelope[object]:
    """Partially update an expansion.

    Only fields present in the request body are modified. Requires
    curator role. Enforces game silo isolation.

    Args:
        game_id: UUID of the parent game.
        expansion_id: UUID of the expansion to update.
        body: Partial update payload with UNSET-aware fields.
        request: Incoming HTTP request (provides request_id).
        db: Database session.

    Returns:
        DataEnvelope containing the updated ExpansionResponse.

    Raises:
        NotFoundError: If the expansion does not exist or does not
            belong to the specified game.
    """
    service = _get_expansion_service(db)
    expansion = await service.update_expansion(game_id, expansion_id, body)
    return ok(_expansion_to_response(expansion), request)


@router.post(
    "/{expansion_id}/archive",
    dependencies=[_curator],
)
async def archive_expansion(
    game_id: uuid.UUID,
    expansion_id: uuid.UUID,
    request: Request,
    db: DbSession,
) -> DataEnvelope[object]:
    """Archive an expansion (soft delete).

    Sets archived_at to the current timestamp. Returns 409 if the
    expansion is already archived. Requires curator role.

    Args:
        game_id: UUID of the parent game.
        expansion_id: UUID of the expansion to archive.
        request: Incoming HTTP request (provides request_id).
        db: Database session.

    Returns:
        DataEnvelope containing the updated ExpansionResponse.

    Raises:
        NotFoundError: If the expansion does not exist or does not
            belong to the specified game.
        ConflictError: If the expansion is already archived.
    """
    service = _get_expansion_service(db)
    expansion = await service.archive_expansion(game_id, expansion_id)
    return ok(_expansion_to_response(expansion), request)


@router.post(
    "/{expansion_id}/restore",
    dependencies=[_curator],
)
async def restore_expansion(
    game_id: uuid.UUID,
    expansion_id: uuid.UUID,
    request: Request,
    db: DbSession,
) -> DataEnvelope[object]:
    """Restore an archived expansion.

    Clears archived_at, making the expansion active again. Returns 409
    if the expansion is not archived. Requires curator role.

    Args:
        game_id: UUID of the parent game.
        expansion_id: UUID of the expansion to restore.
        request: Incoming HTTP request (provides request_id).
        db: Database session.

    Returns:
        DataEnvelope containing the updated ExpansionResponse.

    Raises:
        NotFoundError: If the expansion does not exist or does not
            belong to the specified game.
        ConflictError: If the expansion is not archived.
    """
    service = _get_expansion_service(db)
    expansion = await service.restore_expansion(game_id, expansion_id)
    return ok(_expansion_to_response(expansion), request)
