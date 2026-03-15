"""Game API endpoints — CRUD, search, archive/restore, tags.

All endpoints live under ``/api/v1/games`` (prefix applied in router.py).
Read endpoints require authentication; write endpoints require curator role.
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
from tabletop_oracle.repositories.game_repository import GameRepository
from tabletop_oracle.schemas.common import DataEnvelope, ListEnvelope  # noqa: TC001
from tabletop_oracle.schemas.game import (
    GameCreate,
    GameDetailResponse,
    GameFilters,
    GameResponse,
    GameSummaryResponse,
    GameUpdate,
    TagCountResponse,
)
from tabletop_oracle.services.game_service import GameService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["games"])

#: Annotated dependency for pagination query parameters.
PaginationDep = Annotated[PaginationParams, Depends(get_pagination_params)]

#: Dependency that enforces curator role on write endpoints.
_curator = Depends(require_role("curator"))


def _get_game_service(db: DbSession) -> GameService:
    """Build a GameService wired to the request's database session.

    Args:
        db: Async database session from the request-scoped dependency.

    Returns:
        GameService instance backed by a GameRepository.
    """
    return GameService(GameRepository(db))


def _game_to_response(game: Any) -> GameResponse:
    """Convert a Game ORM entity to a GameResponse schema.

    Extracts tags from the relationship and computes is_active from
    archived_at. Handles both eagerly-loaded tag objects and raw strings.

    Args:
        game: Game ORM entity with tags relationship loaded.

    Returns:
        Populated GameResponse schema.
    """
    tags = sorted(t.tag for t in game.tags) if game.tags else []
    return GameResponse(
        id=game.id,
        created_by=game.created_by,
        name=game.name,
        publisher=game.publisher,
        year_published=game.year_published,
        edition=game.edition,
        min_players=game.min_players,
        max_players=game.max_players,
        description=game.description,
        cover_image_url=game.cover_image_url,
        complexity=game.complexity.value if game.complexity else None,
        tags=tags,
        archived_at=game.archived_at,
        is_active=game.archived_at is None,
        created_at=game.created_at,
        updated_at=game.updated_at,
    )


@router.post(
    "",
    status_code=201,
    dependencies=[_curator],
)
async def create_game(
    body: GameCreate,
    request: Request,
    response: Response,
    db: DbSession,
    user: CurrentUserDep,
) -> DataEnvelope[object]:
    """Create a new game in the catalogue.

    Requires curator role. Returns the created game in the standard
    envelope with a Location header.

    Args:
        body: Validated game creation payload.
        request: Incoming HTTP request (provides request_id).
        response: Outgoing HTTP response (for Location header).
        db: Database session.
        user: Authenticated curator.

    Returns:
        DataEnvelope containing the created GameResponse.
    """
    service = _get_game_service(db)
    game = await service.create_game(body, user.user_id)

    # Re-fetch to get tags loaded on the entity
    game = await service.get_game(game.id)

    game_response = _game_to_response(game)
    response.headers["Location"] = f"/api/v1/games/{game.id}"
    return ok(game_response, request)


@router.get("")
async def list_games(
    request: Request,
    db: DbSession,
    _user: CurrentUserDep,
    pagination: PaginationDep,
    search: str | None = Query(None, description="Full-text search on game name"),
    player_count: int | None = Query(None, gt=0, description="Filter by player count support"),
    complexity: str | None = Query(None, description="Comma-separated complexity values (OR)"),
    tags: str | None = Query(None, description="Comma-separated tags (AND)"),
    archived: bool = Query(False, description="Include archived games"),
    sort: str = Query("name", description="Sort field, prefix with - for descending"),
) -> ListEnvelope[object]:
    """List games with filtering, sorting, and pagination.

    All query parameters are optional. Returns paginated results with
    summary counts (document_count, expansion_count).

    Args:
        request: Incoming HTTP request (provides request_id).
        db: Database session.
        _user: Authenticated user (enforces auth, value unused).
        pagination: Parsed pagination parameters.
        search: Full-text search term.
        player_count: Filter games supporting this player count.
        complexity: Comma-separated complexity values.
        tags: Comma-separated tag values.
        archived: Whether to include archived games.
        sort: Sort field with optional - prefix.

    Returns:
        ListEnvelope containing paginated GameSummaryResponse items.
    """
    filters = GameFilters(
        search=search,
        player_count=player_count,
        complexity=complexity,
        tags=tags,
        archived=archived,
        sort=sort,
    )

    service = _get_game_service(db)
    summaries, total_count = await service.list_games(filters, pagination)

    items = [
        GameSummaryResponse(
            id=s["game"].id,
            created_by=s["game"].created_by,
            name=s["game"].name,
            publisher=s["game"].publisher,
            year_published=s["game"].year_published,
            edition=s["game"].edition,
            min_players=s["game"].min_players,
            max_players=s["game"].max_players,
            description=s["game"].description,
            cover_image_url=s["game"].cover_image_url,
            complexity=s["game"].complexity.value if s["game"].complexity else None,
            tags=s["tags"],
            archived_at=s["game"].archived_at,
            is_active=s["game"].archived_at is None,
            created_at=s["game"].created_at,
            updated_at=s["game"].updated_at,
            document_count=s["document_count"],
            expansion_count=s["expansion_count"],
        )
        for s in summaries
    ]

    items_list: list[object] = list(items)
    return ok_list(items_list, total_count, pagination, request)


@router.get("/tags")
async def list_tags(
    request: Request,
    db: DbSession,
    _user: CurrentUserDep,
) -> DataEnvelope[object]:
    """List all tags with usage counts across active games.

    Results are sorted by count descending, then tag name ascending.
    No pagination — the tag list is expected to be small.

    Args:
        request: Incoming HTTP request (provides request_id).
        db: Database session.
        _user: Authenticated user (enforces auth, value unused).

    Returns:
        DataEnvelope containing list of TagCountResponse items.
    """
    service = _get_game_service(db)
    tag_rows = await service.get_tags()

    tag_responses = [TagCountResponse(tag=row["tag"], count=row["count"]) for row in tag_rows]
    return ok(tag_responses, request)


@router.get("/{game_id}")
async def get_game(
    game_id: uuid.UUID,
    request: Request,
    db: DbSession,
    _user: CurrentUserDep,
) -> DataEnvelope[object]:
    """Get game detail with inline expansions and aggregated counts.

    Returns the full game resource including all expansions (active and
    archived), document count, and expansion count.

    Args:
        game_id: UUID of the game to retrieve.
        request: Incoming HTTP request (provides request_id).
        db: Database session.
        _user: Authenticated user (enforces auth, value unused).

    Returns:
        DataEnvelope containing GameDetailResponse.

    Raises:
        NotFoundError: If the game does not exist.
    """
    from tabletop_oracle.schemas.expansion import ExpansionResponse

    service = _get_game_service(db)
    detail = await service.get_game_detail(game_id)

    game = detail["game"]
    expansion_responses = [
        ExpansionResponse(
            id=exp.id,
            game_id=exp.game_id,
            name=exp.name,
            year_published=exp.year_published,
            description=exp.description,
            archived_at=exp.archived_at,
            created_at=exp.created_at,
            updated_at=exp.updated_at,
        )
        for exp in detail["expansions"]
    ]

    detail_response = GameDetailResponse(
        id=game.id,
        created_by=game.created_by,
        name=game.name,
        publisher=game.publisher,
        year_published=game.year_published,
        edition=game.edition,
        min_players=game.min_players,
        max_players=game.max_players,
        description=game.description,
        cover_image_url=game.cover_image_url,
        complexity=game.complexity.value if game.complexity else None,
        tags=detail["tags"],
        archived_at=game.archived_at,
        is_active=game.archived_at is None,
        created_at=game.created_at,
        updated_at=game.updated_at,
        document_count=detail["document_count"],
        expansion_count=detail["expansion_count"],
        expansions=expansion_responses,
    )

    return ok(detail_response, request)


@router.patch(
    "/{game_id}",
    dependencies=[_curator],
)
async def update_game(
    game_id: uuid.UUID,
    body: GameUpdate,
    request: Request,
    db: DbSession,
) -> DataEnvelope[object]:
    """Partially update a game.

    Only fields present in the request body are modified. Requires
    curator role.

    Args:
        game_id: UUID of the game to update.
        body: Partial update payload with UNSET-aware fields.
        request: Incoming HTTP request (provides request_id).
        db: Database session.

    Returns:
        DataEnvelope containing the updated GameResponse.

    Raises:
        NotFoundError: If the game does not exist.
    """
    service = _get_game_service(db)
    game = await service.update_game(game_id, body)

    # Expire all cached objects to force reload of tags after set_tags
    # (expire on game alone leaves stale GameTag objects in identity map)
    db.expire_all()
    game = await service.get_game(game.id)
    return ok(_game_to_response(game), request)


@router.post(
    "/{game_id}/archive",
    dependencies=[_curator],
)
async def archive_game(
    game_id: uuid.UUID,
    request: Request,
    db: DbSession,
) -> DataEnvelope[object]:
    """Archive a game (soft delete).

    Sets archived_at to the current timestamp. Returns 409 if the
    game is already archived. Requires curator role.

    Args:
        game_id: UUID of the game to archive.
        request: Incoming HTTP request (provides request_id).
        db: Database session.

    Returns:
        DataEnvelope containing the updated GameResponse.

    Raises:
        NotFoundError: If the game does not exist.
        ConflictError: If the game is already archived.
    """
    service = _get_game_service(db)
    game = await service.archive_game(game_id)

    # Re-fetch to get tags
    game = await service.get_game(game.id)
    return ok(_game_to_response(game), request)


@router.post(
    "/{game_id}/restore",
    dependencies=[_curator],
)
async def restore_game(
    game_id: uuid.UUID,
    request: Request,
    db: DbSession,
) -> DataEnvelope[object]:
    """Restore an archived game.

    Clears archived_at, making the game active again. Returns 409 if
    the game is not archived. Requires curator role.

    Args:
        game_id: UUID of the game to restore.
        request: Incoming HTTP request (provides request_id).
        db: Database session.

    Returns:
        DataEnvelope containing the updated GameResponse.

    Raises:
        NotFoundError: If the game does not exist.
        ConflictError: If the game is not archived.
    """
    service = _get_game_service(db)
    game = await service.restore_game(game_id)

    # Re-fetch to get tags
    game = await service.get_game(game.id)
    return ok(_game_to_response(game), request)

