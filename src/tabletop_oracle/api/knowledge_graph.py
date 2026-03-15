"""Curator-facing Knowledge Graph summary API endpoints.

All endpoints live under ``/api/v1/games/{game_id}/knowledge-graph``
(prefix applied in router.py). Curator role is required on all endpoints.
Responses follow the F001 envelope convention.

Design reference: docs/architecture/epic-003-knowledge-graph-engine/design.md
    API Contracts section — Curator KG Summary.
"""

from __future__ import annotations

import logging
import uuid  # noqa: TC003 — FastAPI needs UUID at runtime for path validation
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from tabletop_oracle.api.deps import DbSession  # noqa: TC001
from tabletop_oracle.api.pagination import PaginationParams, get_pagination_params
from tabletop_oracle.api.response import ok, ok_list
from tabletop_oracle.auth.dependencies import require_role
from tabletop_oracle.errors.exceptions import NotFoundError
from tabletop_oracle.schemas.common import DataEnvelope, ListEnvelope  # noqa: TC001
from tabletop_oracle.schemas.knowledge_graph import (
    DocumentContributionResponse,
    DocumentSummary,
    KGSummaryResponse,
    ReprocessResponse,
    SemanticTypeCount,
)
from tabletop_oracle.services.kg.summary import KGSummaryService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["knowledge-graph"])

#: Dependency that enforces curator role on all endpoints.
_curator = Depends(require_role("curator"))

#: Annotated dependency for pagination query parameters.
PaginationDep = Annotated[PaginationParams, Depends(get_pagination_params)]


def _get_summary_service(db: DbSession) -> KGSummaryService:
    """Build a KGSummaryService wired to the request's database session.

    Args:
        db: Async database session from the request-scoped dependency.

    Returns:
        KGSummaryService instance.
    """
    return KGSummaryService(db)


@router.get(
    "/summary",
    dependencies=[_curator],
)
async def get_kg_summary(
    game_id: uuid.UUID,
    request: Request,
    db: DbSession,
) -> DataEnvelope[object]:
    """Get aggregate KG metrics for a game (AC-510).

    Returns concept counts, association counts, semantic type breakdown,
    per-document contribution summaries, and last updated timestamp.
    Requires curator role.

    Args:
        game_id: UUID of the game.
        request: Incoming HTTP request (provides request_id).
        db: Database session.

    Returns:
        DataEnvelope containing KGSummaryResponse.
    """
    service = _get_summary_service(db)
    summary = await service.get_summary(game_id)

    response = KGSummaryResponse(
        game_id=summary.game_id,
        concept_count=summary.concept_count,
        association_count=summary.association_count,
        document_count=summary.document_count,
        semantic_types=[
            SemanticTypeCount(type=st.semantic_type, count=st.count)
            for st in summary.semantic_types
        ],
        documents=[
            DocumentSummary(
                document_id=doc.document_id,
                name=doc.document_name,
                type=doc.document_type,
                concepts_contributed=doc.concepts_contributed,
                associations_contributed=doc.associations_contributed,
            )
            for doc in summary.documents
        ],
        last_updated_at=summary.last_updated_at,
    )
    return ok(response, request)


@router.get(
    "/documents",
    dependencies=[_curator],
)
async def get_kg_documents(
    game_id: uuid.UUID,
    request: Request,
    db: DbSession,
    pagination: PaginationDep,
) -> ListEnvelope[object]:
    """Get per-document contribution detail with pagination.

    Returns documents that have contributed concepts to the KG for
    a game, with concept/association counts and conflict information.
    Requires curator role.

    Args:
        game_id: UUID of the game.
        request: Incoming HTTP request (provides request_id).
        db: Database session.
        pagination: Parsed pagination parameters.

    Returns:
        ListEnvelope containing paginated DocumentContributionResponse items.
    """
    service = _get_summary_service(db)
    contributions, total_count = await service.get_document_contributions(
        game_id,
        limit=pagination.page_size,
        offset=pagination.offset,
    )

    items: list[object] = [
        DocumentContributionResponse(
            document_id=c.document_id,
            name=c.document_name,
            type=c.document_type,
            expansion_id=c.expansion_id,
            concepts_contributed=c.concepts_contributed,
            associations_contributed=c.associations_contributed,
            conflicts_detected=c.conflicts_detected,
            last_processed_at=c.last_processed_at,
        )
        for c in contributions
    ]
    return ok_list(items, total_count, pagination, request)


@router.post(
    "/reprocess",
    status_code=202,
    dependencies=[_curator],
)
async def reprocess_kg(
    game_id: uuid.UUID,
    request: Request,
    db: DbSession,
) -> DataEnvelope[object]:
    """Trigger a full KG rebuild for a game.

    Queues all contributing documents for reprocessing. Returns
    immediately with 202 Accepted and the number of documents queued.
    Requires curator role.

    The actual reprocessing is handled asynchronously by Celery workers
    (not yet implemented — this endpoint returns the count of documents
    that would be queued).

    Args:
        game_id: UUID of the game.
        request: Incoming HTTP request (provides request_id).
        db: Database session.

    Returns:
        DataEnvelope containing ReprocessResponse.

    Raises:
        NotFoundError: If the game has no KG data to reprocess.
    """
    service = _get_summary_service(db)
    documents_queued = await service.count_documents_for_reprocess(game_id)

    if documents_queued == 0:
        raise NotFoundError("Knowledge graph", str(game_id))

    # TODO: Dispatch Celery tasks for actual reprocessing when
    # the worker infrastructure is implemented.
    logger.info("KG reprocess triggered: game=%s documents=%d", game_id, documents_queued)

    response = ReprocessResponse(
        game_id=game_id,
        documents_queued=documents_queued,
    )
    return ok(response, request)
