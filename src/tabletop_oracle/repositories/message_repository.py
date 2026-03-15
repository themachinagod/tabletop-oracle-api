"""Data access for Message entities with eager-loaded relations.

Provides paginated message listing for a session, ordered by sequence,
with citations and context attachments eagerly loaded. Also provides
single-message lookup, cancellation flag setting, sequence assignment,
and creation operations for messages and context attachments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import Select, func, select
from sqlalchemy.orm import selectinload

from tabletop_oracle.models.message import ContextAttachment, Message
from tabletop_oracle.models.session import Session

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession

    from tabletop_oracle.api.pagination import PaginationParams


class MessageRepository:
    """Data access for Message entities within a game session.

    Provides session-scoped paginated queries that eagerly load citations
    and context attachments, single-message retrieval, cancellation
    flag updates, sequence assignment, and creation operations.

    Args:
        session: SQLAlchemy async session for database operations.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_session(self, session_id: uuid.UUID) -> Session | None:
        """Fetch a game session by ID.

        Used by the service layer to validate session existence and
        ownership before querying messages.

        Args:
            session_id: UUID of the game session.

        Returns:
            The Session entity, or None if not found.
        """
        return await self._session.get(Session, session_id)

    async def get_message(self, message_id: uuid.UUID) -> Message | None:
        """Fetch a single message by primary key.

        Args:
            message_id: UUID of the message.

        Returns:
            The Message entity, or None if not found.
        """
        return await self._session.get(Message, message_id)

    async def set_cancelled(
        self,
        message: Message,
        cancelled_at: datetime,
    ) -> Message:
        """Set the cancellation timestamp on a message.

        Args:
            message: The message entity to cancel.
            cancelled_at: The timestamp to record.

        Returns:
            The updated Message entity.
        """
        message.cancelled_at = cancelled_at
        await self._session.flush()
        return message

    async def is_cancelled(self, message_id: uuid.UUID) -> bool:
        """Check whether a message has been cancelled.

        Used by the pipeline's cancel_check callable to detect cancellation
        between stages. Queries only the cancelled_at column for efficiency.

        Args:
            message_id: UUID of the message to check.

        Returns:
            True if the message has a non-NULL cancelled_at value.
        """
        stmt = select(Message.cancelled_at).where(Message.id == message_id)
        result = await self._session.execute(stmt)
        cancelled_at = result.scalar_one_or_none()
        return cancelled_at is not None

    async def get_next_sequence(self, session_id: uuid.UUID) -> int:
        """Return the next sequence number for a session.

        Calculates max(sequence) + 1, or 1 if no messages exist.

        Args:
            session_id: The game session UUID.

        Returns:
            The next sequence number.
        """
        stmt = select(func.coalesce(func.max(Message.sequence), 0)).where(
            Message.session_id == session_id,
        )
        result = await self._session.execute(stmt)
        current_max: int = result.scalar_one()
        return current_max + 1

    async def create_message(self, message: Message) -> Message:
        """Persist a new message and flush to assign server defaults.

        Args:
            message: The Message ORM instance to persist.

        Returns:
            The persisted message with server-generated fields populated.
        """
        self._session.add(message)
        await self._session.flush()
        await self._session.refresh(message)
        return message

    async def create_attachments(
        self,
        attachments: list[ContextAttachment],
    ) -> list[ContextAttachment]:
        """Persist context attachments in bulk.

        Args:
            attachments: List of ContextAttachment ORM instances.

        Returns:
            The persisted attachments with server-generated fields.
        """
        if not attachments:
            return attachments

        for attachment in attachments:
            self._session.add(attachment)
        await self._session.flush()
        for attachment in attachments:
            await self._session.refresh(attachment)
        return attachments

    async def list_by_session(
        self,
        session_id: uuid.UUID,
        pagination: PaginationParams,
        *,
        order: str = "asc",
    ) -> tuple[list[Message], int]:
        """List messages for a session with pagination and eager-loaded relations.

        Citations are loaded for all messages (empty list for non-ai_answer).
        Context attachments are loaded for all messages (empty list for
        non-user_question). The caller can request ascending or descending
        sequence order.

        Args:
            session_id: UUID of the game session.
            pagination: Page number and page size.
            order: Sort direction -- ``"asc"`` or ``"desc"``.

        Returns:
            Tuple of (messages with relations loaded, total count).
        """
        base_query = select(Message).where(Message.session_id == session_id)

        total = await self._count(base_query)

        order_clause = Message.sequence.desc() if order == "desc" else Message.sequence.asc()

        paginated_query = (
            base_query.options(
                selectinload(Message.citations),
                selectinload(Message.context_attachments),
            )
            .order_by(order_clause)
            .offset(pagination.offset)
            .limit(pagination.page_size)
        )

        result = await self._session.execute(paginated_query)
        messages: list[Message] = list(result.scalars().all())

        return messages, total

    async def _count(self, query: Select[Any]) -> int:
        """Execute a COUNT query derived from the given SELECT.

        Args:
            query: Base SELECT statement to count results for.

        Returns:
            Total number of matching rows.
        """
        count_query = select(func.count()).select_from(query.subquery())
        result = await self._session.execute(count_query)
        total: int = result.scalar_one()
        return total
