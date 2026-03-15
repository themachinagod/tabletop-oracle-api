"""Conversation context service for follow-up question support.

Implements a sliding window + lazy summary strategy per ADR-010:
- Short conversations (< threshold): return all messages
- Long conversations: recent messages + LLM-generated summary of older turns
- Summary failure degrades gracefully to a larger sliding window

Context budget allocation (ADR-010):
  KG results (60%) > conversation (30%) > ad-hoc (10%)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from sqlalchemy import select
from sqlalchemy.orm import load_only

from tabletop_oracle.models.message import Message

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (ADR-010)
# ---------------------------------------------------------------------------

#: Messages below this threshold are returned in full (no summary needed).
SUMMARY_THRESHOLD = 20

#: Number of recent messages to include when summarising older turns.
SLIDING_WINDOW_SIZE = 15

#: Expanded window when summary generation fails (graceful degradation).
DEGRADED_WINDOW_SIZE = 30

#: Context budget fractions per ADR-010.
CONTEXT_BUDGET_KG = 0.60
CONTEXT_BUDGET_CONVERSATION = 0.30
CONTEXT_BUDGET_AD_HOC = 0.10

# ---------------------------------------------------------------------------
# Protocol for LLM summary generation
# ---------------------------------------------------------------------------

_SUMMARY_SYSTEM_PROMPT = (
    "You are a concise conversation summariser for a tabletop game rules "
    "assistant. Given a sequence of user questions and AI answers, produce "
    "a brief summary capturing:\n"
    "- Topics discussed\n"
    "- Key rulings or clarifications made\n"
    "- Game state established (if any)\n\n"
    "Be factual and concise. Use bullet points. Do not exceed 200 words."
)


@runtime_checkable
class SummaryGenerator(Protocol):
    """Callable protocol for generating conversation summaries.

    Matches the signature of ``ModelClient.complete`` restricted to the
    fields this service uses. Allows injection of a mock for testing.
    """

    async def __call__(
        self,
        messages: list[dict[str, Any]],
    ) -> str:
        """Generate a summary from the given chat messages.

        Args:
            messages: Chat messages in OpenAI format (role + content dicts).

        Returns:
            Summary text.
        """
        ...


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ConversationContextService:
    """Manages conversation history for follow-up questions.

    Uses a sliding window over recent messages and, for long conversations,
    an LLM-generated summary of older turns. Summary is transient (not
    persisted) and generated lazily on demand.

    Args:
        db: Async SQLAlchemy session for reading messages.
        summary_generator: Async callable that takes OpenAI-format messages
            and returns summary text. Pass ``None`` to disable summaries
            (always uses sliding window).
    """

    def __init__(
        self,
        db: AsyncSession,
        summary_generator: SummaryGenerator | None = None,
    ) -> None:
        self._db = db
        self._summary_generator = summary_generator

    async def get_context(
        self,
        session_id: UUID,
        max_context_messages: int = SUMMARY_THRESHOLD,
    ) -> list[Message]:
        """Select the most relevant prior messages for context.

        Strategy:
        1. If total messages <= ``max_context_messages``, return all.
        2. Otherwise, return the most recent ``SLIDING_WINDOW_SIZE`` messages.
           A summary of older turns is available via ``build_summary``.
        3. If summary generation was requested but failed, the caller can
           fall back to ``DEGRADED_WINDOW_SIZE`` recent messages.

        Args:
            session_id: The game session to retrieve context for.
            max_context_messages: Threshold below which all messages are
                returned. Defaults to ``SUMMARY_THRESHOLD``.

        Returns:
            List of Message ORM objects ordered by sequence (ascending).
        """
        total = await self._count_messages(session_id)

        if total == 0:
            return []

        if total <= max_context_messages:
            return await self._fetch_messages(session_id)

        # Long conversation — return sliding window of recent messages
        return await self._fetch_messages(
            session_id,
            limit=SLIDING_WINDOW_SIZE,
        )

    async def build_summary(
        self,
        session_id: UUID,
        up_to_sequence: int,
    ) -> str | None:
        """Build a summary of conversation turns before the given sequence.

        Uses the ``intent_analysis`` model slot (fast, cheap) via the
        injected summary generator. Returns ``None`` if no summary is
        needed (conversation is short enough) or if summary generation
        is unavailable.

        Args:
            session_id: The game session to summarise.
            up_to_sequence: Summarise messages with sequence < this value.

        Returns:
            Summary text, or ``None`` if not needed or generation fails.
        """
        older_messages = await self._fetch_messages(
            session_id,
            before_sequence=up_to_sequence,
        )

        if not older_messages:
            return None

        if self._summary_generator is None:
            logger.debug(
                "Summary generator not configured; skipping summary",
                extra={"session_id": str(session_id)},
            )
            return None

        try:
            return await self._generate_summary(older_messages)
        except Exception:
            logger.warning(
                "Summary generation failed; degrading to larger window",
                extra={"session_id": str(session_id)},
                exc_info=True,
            )
            return None

    async def get_context_with_summary(
        self,
        session_id: UUID,
        max_context_messages: int = SUMMARY_THRESHOLD,
    ) -> tuple[list[Message], str | None]:
        """Retrieve context messages and optional summary in one call.

        Convenience method combining ``get_context`` and ``build_summary``.
        If the conversation is short, returns all messages and no summary.
        If long, returns the sliding window and attempts a summary. On
        summary failure, expands the window to ``DEGRADED_WINDOW_SIZE``.

        Args:
            session_id: The game session to retrieve context for.
            max_context_messages: Threshold for short vs. long conversation.

        Returns:
            Tuple of (messages, summary_or_none).
        """
        total = await self._count_messages(session_id)

        if total == 0:
            return [], None

        if total <= max_context_messages:
            messages = await self._fetch_messages(session_id)
            return messages, None

        # Long conversation — get sliding window
        recent = await self._fetch_messages(
            session_id,
            limit=SLIDING_WINDOW_SIZE,
        )

        if not recent:
            return [], None

        # The oldest message in the window tells us where to summarise up to
        oldest_in_window = recent[0].sequence
        summary = await self.build_summary(session_id, up_to_sequence=oldest_in_window)

        if summary is None and self._summary_generator is not None:
            # Summary failed — degrade to larger window
            recent = await self._fetch_messages(
                session_id,
                limit=DEGRADED_WINDOW_SIZE,
            )

        return recent, summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _count_messages(self, session_id: UUID) -> int:
        """Count total messages in a session.

        Args:
            session_id: Session to count messages for.

        Returns:
            Total message count.
        """
        from sqlalchemy import func as sa_func

        stmt = (
            select(sa_func.count())
            .select_from(Message)
            .where(
                Message.session_id == session_id,
            )
        )
        result = await self._db.execute(stmt)
        return result.scalar_one()

    async def _fetch_messages(
        self,
        session_id: UUID,
        *,
        limit: int | None = None,
        before_sequence: int | None = None,
    ) -> list[Message]:
        """Fetch messages from the database, ordered by sequence ascending.

        Args:
            session_id: Session to fetch messages for.
            limit: Maximum number of messages to return (most recent N).
            before_sequence: Only include messages with sequence < this value.

        Returns:
            List of Message ORM objects ordered by sequence ascending.
        """
        stmt = (
            select(Message)
            .options(
                load_only(
                    Message.id,
                    Message.session_id,
                    Message.type,
                    Message.content,
                    Message.sequence,
                    Message.created_at,
                ),
            )
            .where(Message.session_id == session_id)
        )

        if before_sequence is not None:
            stmt = stmt.where(Message.sequence < before_sequence)

        stmt = stmt.order_by(Message.sequence.asc())

        if limit is not None:
            # We want the N most recent, but ordered ascending.
            # Use a subquery: get the last N by desc, then re-order asc.
            desc_stmt = (
                select(Message)
                .options(
                    load_only(
                        Message.id,
                        Message.session_id,
                        Message.type,
                        Message.content,
                        Message.sequence,
                        Message.created_at,
                    ),
                )
                .where(Message.session_id == session_id)
            )
            if before_sequence is not None:
                desc_stmt = desc_stmt.where(Message.sequence < before_sequence)

            desc_stmt = desc_stmt.order_by(Message.sequence.desc()).limit(limit)

            # Wrap in subquery and re-order ascending
            subq = desc_stmt.subquery()
            alias = (
                select(Message)
                .join(subq, Message.id == subq.c.id)
                .order_by(Message.sequence.asc())
            )
            result = await self._db.execute(alias)
            return list(result.scalars().all())

        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    async def _generate_summary(self, messages: list[Message]) -> str:
        """Format messages and call the summary generator.

        Args:
            messages: Messages to summarise (ordered by sequence).

        Returns:
            Generated summary text.

        Raises:
            Exception: Propagated from the summary generator.
        """
        assert self._summary_generator is not None

        chat_messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
        ]

        for msg in messages:
            role = "user" if msg.type.value == "user_question" else "assistant"
            chat_messages.append({"role": role, "content": msg.content})

        return await self._summary_generator(messages=chat_messages)
