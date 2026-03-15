"""Message API endpoints -- history and cancellation.

Provides ``GET /api/v1/sessions/{session_id}/messages`` for paginated
message history, and ``DELETE /api/v1/sessions/{session_id}/messages/{message_id}``
for query cancellation. Pagination, sort order, session ownership
validation, and the standard F001 response envelope are applied throughout.
"""

from __future__ import annotations

import logging
import uuid  # noqa: TC003 -- FastAPI needs UUID at runtime for path validation
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request, Response

from tabletop_oracle.api.deps import DbSession  # noqa: TC001
from tabletop_oracle.api.pagination import PaginationParams, get_pagination_params
from tabletop_oracle.api.response import ok, ok_list
from tabletop_oracle.auth.dependencies import CurrentUserDep  # noqa: TC001
from tabletop_oracle.repositories.message_repository import MessageRepository
from tabletop_oracle.schemas.common import DataEnvelope, ListEnvelope  # noqa: TC001
from tabletop_oracle.schemas.message import (
    CancelResponse,
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


@router.delete("/{session_id}/messages/{message_id}")
async def cancel_message(
    session_id: uuid.UUID,
    message_id: uuid.UUID,
    request: Request,
    response: Response,
    db: DbSession,
    user: CurrentUserDep,
) -> DataEnvelope[object] | None:
    """Cancel an in-progress query.

    Sets the cancellation flag on the message record. The pipeline's
    cancel_check detects this flag between stages and emits a
    ``stream.cancelled`` SSE event.

    Returns 202 Accepted if the message was actively processing and
    cancellation was initiated. Returns 204 No Content if the message
    was already complete or already cancelled.

    Args:
        session_id: UUID of the game session.
        message_id: UUID of the message to cancel.
        request: Incoming HTTP request (provides request_id).
        response: HTTP response (for setting status code).
        db: Database session.
        user: Authenticated user (session ownership enforced).

    Returns:
        DataEnvelope with CancelResponse on 202, or None on 204.

    Raises:
        NotFoundError: If the session or message does not exist, the
            user does not own the session, or the message does not
            belong to the session.
    """
    service = _get_message_service(db)
    message, was_cancelled = await service.cancel_message(
        session_id,
        message_id,
        user,
    )

    if not was_cancelled:
        response.status_code = 204
        return None

    response.status_code = 202
    cancel_data = CancelResponse(id=message.id, status="cancelling")
    return ok(cancel_data, request)


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
        description=("Sort order by sequence: 'asc' (oldest first) or 'desc' (newest first)"),
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
        order: Sort direction -- ``"asc"`` or ``"desc"``.

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
        cancelled_at=message.cancelled_at,
        citations=citations,
        context_attachments=context_attachments,
    )
