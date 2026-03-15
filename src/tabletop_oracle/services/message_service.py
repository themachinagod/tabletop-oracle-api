"""Business logic for message submission, history retrieval, and query cancellation.

Encapsulates session ownership validation, message persistence with
sequence numbering, context attachment handling, delegates paginated message
listing to the MessageRepository, and handles query cancellation by
setting the cancelled_at flag on in-progress messages. All domain errors
use the F001 exception hierarchy.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from tabletop_oracle.auth.ownership import check_ownership
from tabletop_oracle.errors.exceptions import (
    ConflictError,
    ContentTooLargeError,
    NotFoundError,
    UnprocessableEntityError,
)
from tabletop_oracle.models.enums import (
    ContextAttachmentType,
    MessageType,
    SessionStatus,
)
from tabletop_oracle.models.message import ContextAttachment
from tabletop_oracle.models.message import Message as MessageModel

if TYPE_CHECKING:
    import uuid

    from tabletop_oracle.api.pagination import PaginationParams
    from tabletop_oracle.auth.models import CurrentUser
    from tabletop_oracle.models.message import Message
    from tabletop_oracle.repositories.message_repository import MessageRepository
    from tabletop_oracle.schemas.message import MessageSubmitRequest

logger = logging.getLogger(__name__)

#: Maximum image file size in bytes (10 MB).
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024


class MessageService:
    """Business logic for message submission, history retrieval, and cancellation.

    Validates session existence and ownership before delegating to the
    repository for paginated message retrieval with eager-loaded relations,
    message submission with sequence numbering, and query cancellation.

    Args:
        message_repo: Repository for message data access operations.
    """

    def __init__(self, message_repo: MessageRepository) -> None:
        self._repo = message_repo

    async def submit_message(
        self,
        session_id: uuid.UUID,
        request: MessageSubmitRequest,
        current_user: CurrentUser,
        image_files: dict[str, tuple[bytes, str]] | None = None,
    ) -> Message:
        """Submit a player message to a session.

        Validates ownership/status, assigns sequence, persists message
        and attachments.

        Args:
            session_id: The game session UUID.
            request: Validated submission request.
            current_user: The authenticated user.
            image_files: file_name -> (bytes, content_type) for images.

        Returns:
            The persisted Message with context_attachments loaded.

        Raises:
            NotFoundError: Session not found or not owned.
            ConflictError: Session is archived.
            UnprocessableEntityError: Content is whitespace-only.
            ContentTooLargeError: Image exceeds 10MB.
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

        if session.status == SessionStatus.ARCHIVED:
            raise ConflictError("Session is archived")

        if not request.content.strip():
            raise UnprocessableEntityError("Message content must not be empty")

        if image_files:
            for file_name, (data, _ct) in image_files.items():
                if len(data) > MAX_IMAGE_SIZE_BYTES:
                    raise ContentTooLargeError(f"Image '{file_name}' exceeds maximum size of 10MB")

        sequence = await self._repo.get_next_sequence(session_id)
        message = MessageModel(
            session_id=session_id,
            type=MessageType.USER_QUESTION,
            content=request.content,
            sequence=sequence,
        )
        message = await self._repo.create_message(message)

        attachments = await self._build_attachments(
            message,
            session_id,
            request,
            image_files,
        )
        message.context_attachments = attachments

        logger.info(
            "Message submitted",
            extra={
                "message_id": str(message.id),
                "session_id": str(session_id),
                "sequence": sequence,
                "attachment_count": len(attachments),
            },
        )
        return message

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

    async def _build_attachments(
        self,
        message: Message,
        session_id: uuid.UUID,
        request: MessageSubmitRequest,
        image_files: dict[str, tuple[bytes, str]] | None,
    ) -> list[ContextAttachment]:
        """Build and persist context attachments."""
        attachments: list[ContextAttachment] = []
        for att_req in request.context_attachments:
            if att_req.type == "text":
                attachments.append(
                    ContextAttachment(
                        message_id=message.id,
                        type=ContextAttachmentType.TEXT,
                        content=att_req.content,
                    )
                )
            elif att_req.type == "image":
                fp, fs = None, None
                if image_files and att_req.file_name and att_req.file_name in image_files:
                    data, _ = image_files[att_req.file_name]
                    fs = len(data)
                    fp = f"sessions/{session_id}/messages/{message.id}/{att_req.file_name}"
                attachments.append(
                    ContextAttachment(
                        message_id=message.id,
                        type=ContextAttachmentType.IMAGE,
                        file_name=att_req.file_name,
                        file_path=fp,
                        file_size=fs,
                    )
                )
        if attachments:
            await self._repo.create_attachments(attachments)
        return attachments

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
            order: Sort direction -- ``"asc"`` or ``"desc"``.

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
