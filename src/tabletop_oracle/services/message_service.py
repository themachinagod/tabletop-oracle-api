"""Business logic for message history retrieval and query cancellation.

Encapsulates session ownership validation, delegates paginated message
listing to the MessageRepository, and handles query cancellation by
setting the cancelled_at flag on in-progress messages. All domain errors
use the F001 exception hierarchy.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from tabletop_oracle.auth.ownership import check_ownership
from tabletop_oracle.errors.exceptions import NotFoundError

if TYPE_CHECKING:
    import uuid

    from tabletop_oracle.api.pagination import PaginationParams
    from tabletop_oracle.auth.models import CurrentUser
    from tabletop_oracle.models.message import Message
    from tabletop_oracle.repositories.message_repository import MessageRepository

logger = logging.getLogger(__name__)


class MessageService:
    """Business logic for retrieving conversation message history and cancellation.

    Validates session existence and ownership before delegating to the
    repository for paginated message retrieval with eager-loaded relations,
    and handles query cancellation.

    Args:
        message_repo: Repository for message data access operations.
    """

    def __init__(self, message_repo: MessageRepository) -> None:
        self._repo = message_repo

    async def cancel_message(
        self,
        session_id: uuid.UUID,
        message_id: uuid.UUID,
        current_user: CurrentUser,
    ) -> tuple[Message, bool]:
        """Cancel an in-progress query.

        Validates session ownership and message existence. If the message
        is still being processed (no confidence_score yet and not already
        cancelled), sets the cancelled_at timestamp and returns (message, True).
        If the message is already complete or already cancelled, returns
        (message, False) indicating no cancellation was performed.

        Args:
            session_id: UUID of the game session owning the message.
            message_id: UUID of the message to cancel.
            current_user: The authenticated user making the request.

        Returns:
            Tuple of (message, was_cancelled). was_cancelled is True if
            the cancellation flag was newly set, False if the message
            was already complete or already cancelled.

        Raises:
            NotFoundError: If the session or message does not exist, the
                user does not own the session, or the message does not
                belong to the session.
        """
        session = await self._repo.get_session(session_id)
        if session is None:
            raise NotFoundError("Session", str(session_id))

        check_ownership(
            resource_user_id=session.user_id,
            current_user=current_user,
            resource_type="Session",
            resource_id=str(session_id),
        )

        message = await self._repo.get_message(message_id)
        if message is None or message.session_id != session_id:
            raise NotFoundError("Message", str(message_id))

        # Already cancelled or already complete -- no action needed
        if message.cancelled_at is not None:
            return message, False

        if message.confidence_score is not None:
            return message, False

        await self._repo.set_cancelled(message, datetime.now(UTC))

        logger.info(
            "Message cancelled",
            extra={
                "message_id": str(message_id),
                "session_id": str(session_id),
            },
        )

        return message, True

    async def list_messages(
        self,
        session_id: uuid.UUID,
        current_user: CurrentUser,
        pagination: PaginationParams,
        *,
        order: str = "asc",
    ) -> tuple[list[Message], int]:
        """List messages for a session with ownership validation.

        Validates that the session exists and belongs to the current user
        (or the user is a curator). Returns paginated messages ordered
        by sequence with citations and context attachments loaded.

        Args:
            session_id: UUID of the game session.
            current_user: The authenticated user making the request.
            pagination: Page number and page size.
            order: Sort direction for sequence -- ``"asc"`` or ``"desc"``.

        Returns:
            Tuple of (messages with relations, total count).

        Raises:
            NotFoundError: If the session does not exist or the user
                does not own it (returns 404 to avoid leaking existence).
        """
        session = await self._repo.get_session(session_id)
        if session is None:
            raise NotFoundError("Session", str(session_id))

        check_ownership(
            resource_user_id=session.user_id,
            current_user=current_user,
            resource_type="Session",
            resource_id=str(session_id),
        )

        return await self._repo.list_by_session(
            session_id,
            pagination,
            order=order,
        )
