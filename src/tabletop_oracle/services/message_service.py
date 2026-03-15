"""Business logic for message history retrieval.

Encapsulates session ownership validation and delegates paginated message
listing to the MessageRepository. All domain errors use the F001 exception
hierarchy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tabletop_oracle.auth.ownership import check_ownership
from tabletop_oracle.errors.exceptions import NotFoundError

if TYPE_CHECKING:
    import uuid

    from tabletop_oracle.api.pagination import PaginationParams
    from tabletop_oracle.auth.models import CurrentUser
    from tabletop_oracle.models.message import Message
    from tabletop_oracle.repositories.message_repository import MessageRepository


class MessageService:
    """Business logic for retrieving conversation message history.

    Validates session existence and ownership before delegating to the
    repository for paginated message retrieval with eager-loaded relations.

    Args:
        message_repo: Repository for message data access operations.
    """

    def __init__(self, message_repo: MessageRepository) -> None:
        self._repo = message_repo

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
            order: Sort direction for sequence — ``"asc"`` or ``"desc"``.

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
