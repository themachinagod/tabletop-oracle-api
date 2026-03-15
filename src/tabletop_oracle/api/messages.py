"""Message history API endpoint — paginated GET for session messages.

Provides ``GET /api/v1/sessions/{session_id}/messages`` with pagination,
sort order, session ownership validation, and the standard F001 response
envelope. Citations are included for ai_answer messages; context attachments
for user_question messages.
"""

from __future__ import annotations

import logging
import uuid  # noqa: TC003 — FastAPI needs UUID at runtime for path validation
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request

from tabletop_oracle.api.deps import DbSession  # noqa: TC001
from tabletop_oracle.api.pagination import PaginationParams, get_pagination_params
from tabletop_oracle.api.response import ok_list
from tabletop_oracle.auth.dependencies import CurrentUserDep  # noqa: TC001
from tabletop_oracle.repositories.message_repository import MessageRepository
from tabletop_oracle.schemas.common import ListEnvelope  # noqa: TC001
from tabletop_oracle.schemas.message import (
    CitationResponse,
    ContextAttachmentResponse,
    MessageResponse,
)
from tabletop_oracle.services.message_service import MessageService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["messages"])

#: Annotated dependency for pagination query parameters.
PaginationDep = Annotated[PaginationParams, Depends(get_pagination_params)]


def _get_message_service(db: DbSession) -> MessageService:
    """Build a MessageService wired to the request's database session.

    Args:
        db: Async database session from the request-scoped dependency.

    Returns:
        MessageService instance backed by a MessageRepository.
    """
    return MessageService(MessageRepository(db))


@router.get("/{session_id}/messages")
async def list_messages(
    session_id: uuid.UUID,
    request: Request,
    db: DbSession,
    user: CurrentUserDep,
    pagination: PaginationDep,
    order: str = Query(
        default="asc",
        pattern="^(asc|desc)$",
        description="Sort order by sequence: 'asc' (oldest first) or 'desc' (newest first)",
    ),
) -> ListEnvelope[object]:
    """List messages for a game session with pagination.

    Returns messages ordered by sequence number. Citations are included
    for ai_answer messages. Context attachments are included for
    user_question messages. Requires session ownership or curator role.

    Args:
        session_id: UUID of the game session.
        request: Incoming HTTP request (provides request_id).
        db: Database session.
        user: Authenticated user (session ownership enforced).
        pagination: Parsed pagination parameters.
        order: Sort direction — ``"asc"`` or ``"desc"``.

    Returns:
        ListEnvelope containing paginated MessageResponse items.

    Raises:
        NotFoundError: If the session does not exist or the user
            does not own it.
    """
    service = _get_message_service(db)
    messages, total_count = await service.list_messages(
        session_id,
        user,
        pagination,
        order=order,
    )

    items: list[object] = [_message_to_response(msg) for msg in messages]
    return ok_list(items, total_count, pagination, request)


def _message_to_response(message: Any) -> MessageResponse:
    """Convert a Message ORM entity to a MessageResponse schema.

    Maps citations and context attachments from the ORM relationships
    to their respective response schemas.

    Args:
        message: Message ORM entity with relations loaded.

    Returns:
        Populated MessageResponse schema.
    """
    citations = [
        CitationResponse(
            id=c.id,
            document_id=c.document_id,
            document_name=c.document_name,
            document_type=c.document_type.value,
            section_path=c.section_path,
            page_number=c.page_number,
            excerpt=c.excerpt,
        )
        for c in message.citations
    ]

    context_attachments = [
        ContextAttachmentResponse(
            id=a.id,
            type=a.type.value,
            content=a.content,
            file_path=a.file_path,
            file_name=a.file_name,
            file_size=a.file_size,
            processed_description=a.processed_description,
        )
        for a in message.context_attachments
    ]

    return MessageResponse(
        id=message.id,
        session_id=message.session_id,
        type=message.type.value,
        content=message.content,
        sequence=message.sequence,
        in_reply_to_id=message.in_reply_to_id,
        confidence_score=message.confidence_score,
        processing_duration_ms=message.processing_duration_ms,
        created_at=message.created_at,
        citations=citations,
        context_attachments=context_attachments,
    )
