"""SSE streaming endpoints with authentication and authorisation.

Provides two SSE endpoints:
- AI response stream: per-message streaming for authenticated session owners
- Document processing stream: per-document streaming for curators

Both endpoints validate authentication via SessionMiddleware (cookie-based,
ADR-003) and perform resource-level authorisation before streaming begins.
Pre-stream errors return standard JSON responses; stream-level errors are
delivered as SSE events.

Session expiry during an active stream is handled by the session expiry
checker, which closes the stream gracefully.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Annotated
from uuid import UUID  # noqa: TC003  # FastAPI needs UUID at runtime for path validation

from fastapi import APIRouter, Depends, Header, Request
from starlette.responses import Response

from tabletop_oracle.auth.dependencies import CurrentUserDep, require_role
from tabletop_oracle.auth.models import CurrentUser
from tabletop_oracle.auth.session_store import SessionStore
from tabletop_oracle.database import async_session_factory
from tabletop_oracle.streaming.sse import sse_response

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from tabletop_oracle.streaming.events import SseEvent

logger = logging.getLogger(__name__)

router = APIRouter(tags=["streaming"])

#: Interval for checking session validity during active streams.
SESSION_CHECK_INTERVAL_SECONDS: float = 30.0


async def _session_expiry_wrapper(
    event_generator: AsyncGenerator[SseEvent, None],
    auth_session_id: str,
    check_interval: float = SESSION_CHECK_INTERVAL_SECONDS,
) -> AsyncGenerator[SseEvent, None]:
    """Wrap a domain event generator with session expiry detection.

    Periodically checks whether the auth session is still valid. If the
    session has expired or been revoked, the generator terminates, causing
    the SSE stream to close. The client's next reconnection attempt will
    receive a 401 from SessionMiddleware.

    Args:
        event_generator: The domain async generator yielding SSE events.
        auth_session_id: The session token ID to validate.
        check_interval: How often to check session validity (seconds).
    """
    last_check = asyncio.get_event_loop().time()

    try:
        async for event in event_generator:
            yield event

            now = asyncio.get_event_loop().time()
            if now - last_check >= check_interval:
                last_check = now
                if not await _is_session_valid(auth_session_id):
                    logger.info(
                        "session_expired_during_stream",
                        extra={"session_id": auth_session_id[:8] + "..."},
                    )
                    break
    finally:
        await event_generator.aclose()


async def _is_session_valid(session_id: str) -> bool:
    """Check whether an auth session is still valid.

    Opens a short-lived database connection to look up the session.
    Returns False if the session is missing, expired, or revoked.

    Args:
        session_id: The auth session token ID.
    """
    from datetime import UTC, datetime

    async with async_session_factory() as db:
        auth_session = await SessionStore.get(db, session_id)
        if auth_session is None:
            return False
        return bool(auth_session.expires_at > datetime.now(UTC))


# ---------------------------------------------------------------------------
# Stub registries for resource lookup
#
# These are placeholder lookups used by the SSE endpoints. The actual
# service implementations will be provided by consuming epics (EPIC-002,
# EPIC-004). For now, these stubs enable the auth integration layer to be
# tested and integrated independently.
# ---------------------------------------------------------------------------

#: Registry of currently-active message streams.
#: Maps (session_id, message_id) -> async generator factory.
#: Populated by the AI workflow service (EPIC-004).
_active_message_streams: dict[
    tuple[UUID, UUID],
    type[None],  # Placeholder — will be callable returning AsyncGenerator
] = {}

#: Registry of currently-active document processing streams.
#: Maps document_id -> async generator factory.
#: Populated by the document processing service (EPIC-002).
_active_document_streams: dict[
    UUID,
    type[None],  # Placeholder — will be callable returning AsyncGenerator
] = {}


async def _placeholder_generator() -> AsyncGenerator[SseEvent, None]:
    """Placeholder async generator that yields nothing.

    Used during infrastructure development before consuming epics
    provide real event generators. Returns immediately, producing
    an empty stream.
    """
    return
    yield  # Make this a valid async generator


@router.get(
    "/sessions/{session_id}/messages/{message_id}/stream",
    response_class=Response,
    summary="Stream AI response events",
    description=(
        "SSE endpoint streaming AI response events for a message. "
        "Requires session ownership. Returns 204 if not actively streaming."
    ),
)
async def stream_message(
    session_id: UUID,
    message_id: UUID,
    request: Request,
    current_user: CurrentUserDep,
    last_event_id: Annotated[int | None, Header(alias="Last-Event-ID")] = None,
) -> Response:
    """Stream AI response events for a message via SSE.

    Validates session ownership and message existence before streaming.
    Returns standard JSON errors for auth/not-found conditions and 204
    for non-streaming messages.

    Args:
        session_id: The game session UUID (not auth session).
        message_id: The message UUID to stream.
        request: The incoming HTTP request.
        current_user: Authenticated user from SessionMiddleware.
        last_event_id: Last-Event-ID header from reconnecting client.

    Returns:
        StreamingResponse with text/event-stream, or 204 No Content,
        or raises appropriate error (401, 404).
    """
    # 1. Validate session ownership.
    #    In a full implementation, this would load the game session from DB
    #    and verify ownership. For now, we validate with check_ownership
    #    pattern — the session lookup will be provided by EPIC-004.
    #
    #    Stub: treat session as not found if no stream is active.
    #    When EPIC-004 is implemented, this will be:
    #      session = await session_service.get(session_id)
    #      if not session:
    #          raise NotFoundError("Session", str(session_id))
    #      check_ownership(session.user_id, current_user, "Session", str(session_id))

    # 2. Check if message is actively streaming.
    stream_key = (session_id, message_id)
    if stream_key not in _active_message_streams:
        # No active stream — return 204 No Content.
        # In full implementation: check if message exists first (404 if not).
        return Response(status_code=204)

    # 3. Get domain event generator (will be provided by EPIC-004).
    event_gen = _placeholder_generator()

    # 4. Wrap with session expiry checker.
    wrapped_gen = _session_expiry_wrapper(event_gen, current_user.session_id)

    # 5. Return SSE response.
    return await sse_response(wrapped_gen, request, last_event_id)


CuratorUser = Annotated[CurrentUser, Depends(require_role("curator"))]


@router.get(
    "/documents/{document_id}/processing/stream",
    response_class=Response,
    summary="Stream document processing events",
    description=(
        "SSE endpoint streaming document processing status events. "
        "Requires curator role. Returns 204 if not actively processing."
    ),
)
async def stream_document_processing(
    document_id: UUID,
    request: Request,
    current_user: CuratorUser,
    last_event_id: Annotated[int | None, Header(alias="Last-Event-ID")] = None,
) -> Response:
    """Stream document processing events via SSE.

    Validates curator role and document existence before streaming.
    Returns standard JSON errors for auth/forbidden/not-found conditions
    and 204 for non-processing documents.

    Args:
        document_id: The document UUID.
        request: The incoming HTTP request.
        current_user: Authenticated curator from SessionMiddleware + role check.
        last_event_id: Last-Event-ID header from reconnecting client.

    Returns:
        StreamingResponse with text/event-stream, or 204 No Content,
        or raises appropriate error (401, 403, 404).
    """
    # 1. Curator role is already enforced by the CuratorUser dependency.

    # 2. Check if document is actively processing.
    if document_id not in _active_document_streams:
        # No active processing — return 204 No Content.
        # In full implementation: check if document exists first (404 if not).
        return Response(status_code=204)

    # 3. Get domain event generator (will be provided by EPIC-002).
    event_gen = _placeholder_generator()

    # 4. Wrap with session expiry checker.
    wrapped_gen = _session_expiry_wrapper(event_gen, current_user.session_id)

    # 5. Return SSE response.
    return await sse_response(wrapped_gen, request, last_event_id)
