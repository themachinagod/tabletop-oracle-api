"""Unit tests for ConversationContextService.

Tests the sliding window + lazy summary strategy per ADR-010:
- Short conversations return all messages
- Long conversations return sliding window + summary
- Summary failure degrades to larger window
- Empty conversation returns empty list
- Boundary cases at threshold edges
- Messages ordered correctly by sequence
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tabletop_oracle.models.enums import MessageType
from tabletop_oracle.services.ai.conversation import (
    DEGRADED_WINDOW_SIZE,
    SLIDING_WINDOW_SIZE,
    SUMMARY_THRESHOLD,
    ConversationContextService,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SESSION_ID = uuid.uuid4()


def _make_message(
    sequence: int,
    *,
    session_id: uuid.UUID = _SESSION_ID,
    msg_type: MessageType = MessageType.USER_QUESTION,
    content: str | None = None,
) -> MagicMock:
    """Build a mock Message ORM object.

    Args:
        sequence: Message sequence number.
        session_id: Owning session ID.
        msg_type: Message type enum value.
        content: Message text (defaults to "Message {sequence}").

    Returns:
        MagicMock mimicking a Message ORM instance.
    """
    mock = MagicMock()
    mock.id = uuid.uuid4()
    mock.session_id = session_id
    mock.type = msg_type
    mock.content = content or f"Message {sequence}"
    mock.sequence = sequence
    mock.created_at = datetime.now(tz=UTC)
    return mock


def _make_messages(count: int, *, start_seq: int = 1) -> list[MagicMock]:
    """Build a list of mock messages with alternating user/assistant types.

    Args:
        count: Number of messages to create.
        start_seq: Starting sequence number.

    Returns:
        List of mock Message objects ordered by sequence ascending.
    """
    messages = []
    for i in range(count):
        seq = start_seq + i
        msg_type = MessageType.USER_QUESTION if i % 2 == 0 else MessageType.AI_ANSWER
        messages.append(_make_message(seq, msg_type=msg_type))
    return messages


def _mock_db(messages: list[MagicMock], count: int | None = None) -> AsyncMock:
    """Build a mock AsyncSession that returns the given messages.

    The mock handles both count queries and message fetch queries.
    Count queries return ``count`` (defaults to ``len(messages)``).
    Fetch queries return the messages list.

    Args:
        messages: Messages to return from fetch queries.
        count: Total count to report (defaults to len(messages)).

    Returns:
        AsyncMock mimicking an AsyncSession.
    """
    if count is None:
        count = len(messages)

    db = AsyncMock()

    def make_count_result() -> MagicMock:
        result = MagicMock()
        result.scalar_one.return_value = count
        return result

    def make_fetch_result(msgs: list[MagicMock]) -> MagicMock:
        result = MagicMock()
        result.scalars.return_value.all.return_value = msgs
        return result

    # We need to handle multiple execute calls:
    # 1st call is always count, subsequent are fetches
    call_count_tracker = {"n": 0}
    original_messages = list(messages)

    async def execute_side_effect(stmt: Any) -> MagicMock:
        call_count_tracker["n"] += 1
        # The count query uses func.count(), detectable by checking
        # if the statement selects from Message or count(*)
        stmt_str = str(stmt)
        if "count" in stmt_str.lower():
            return make_count_result()
        return make_fetch_result(original_messages)

    db.execute = AsyncMock(side_effect=execute_side_effect)
    return db


# ---------------------------------------------------------------------------
# Empty conversation
# ---------------------------------------------------------------------------


class TestEmptyConversation:
    """Tests for sessions with no messages."""

    @pytest.mark.asyncio
    async def test_get_context_returns_empty_list(self) -> None:
        """Empty session returns an empty message list."""
        db = _mock_db([], count=0)
        service = ConversationContextService(db=db)

        result = await service.get_context(_SESSION_ID)

        assert result == []

    @pytest.mark.asyncio
    async def test_get_context_with_summary_returns_empty(self) -> None:
        """Empty session returns empty list and no summary."""
        db = _mock_db([], count=0)
        service = ConversationContextService(db=db)

        messages, summary = await service.get_context_with_summary(_SESSION_ID)

        assert messages == []
        assert summary is None


# ---------------------------------------------------------------------------
# Short conversation (below threshold)
# ---------------------------------------------------------------------------


class TestShortConversation:
    """Tests for conversations below SUMMARY_THRESHOLD."""

    @pytest.mark.asyncio
    async def test_returns_all_messages_when_below_threshold(self) -> None:
        """All messages returned when count < SUMMARY_THRESHOLD."""
        msgs = _make_messages(10)
        db = _mock_db(msgs, count=10)
        service = ConversationContextService(db=db)

        result = await service.get_context(_SESSION_ID)

        assert len(result) == 10

    @pytest.mark.asyncio
    async def test_returns_all_messages_at_threshold_minus_one(self) -> None:
        """Boundary: 19 messages (threshold - 1) returns all."""
        msgs = _make_messages(SUMMARY_THRESHOLD - 1)
        db = _mock_db(msgs, count=SUMMARY_THRESHOLD - 1)
        service = ConversationContextService(db=db)

        result = await service.get_context(_SESSION_ID)

        assert len(result) == SUMMARY_THRESHOLD - 1

    @pytest.mark.asyncio
    async def test_returns_all_at_exact_threshold(self) -> None:
        """Boundary: exactly SUMMARY_THRESHOLD messages returns all."""
        msgs = _make_messages(SUMMARY_THRESHOLD)
        db = _mock_db(msgs, count=SUMMARY_THRESHOLD)
        service = ConversationContextService(db=db)

        result = await service.get_context(_SESSION_ID)

        assert len(result) == SUMMARY_THRESHOLD

    @pytest.mark.asyncio
    async def test_short_conversation_no_summary(self) -> None:
        """Short conversations produce no summary via get_context_with_summary."""
        msgs = _make_messages(10)
        db = _mock_db(msgs, count=10)
        service = ConversationContextService(db=db)

        messages, summary = await service.get_context_with_summary(_SESSION_ID)

        assert len(messages) == 10
        assert summary is None


# ---------------------------------------------------------------------------
# Long conversation (above threshold)
# ---------------------------------------------------------------------------


class TestLongConversation:
    """Tests for conversations exceeding SUMMARY_THRESHOLD."""

    @pytest.mark.asyncio
    async def test_returns_sliding_window_when_above_threshold(self) -> None:
        """Only SLIDING_WINDOW_SIZE recent messages returned for long conversations."""
        recent = _make_messages(SLIDING_WINDOW_SIZE, start_seq=6)
        db = _mock_db(recent, count=SUMMARY_THRESHOLD + 1)
        service = ConversationContextService(db=db)

        result = await service.get_context(_SESSION_ID)

        assert len(result) == SLIDING_WINDOW_SIZE

    @pytest.mark.asyncio
    async def test_boundary_at_threshold_plus_one(self) -> None:
        """Boundary: 21 messages triggers sliding window."""
        recent = _make_messages(SLIDING_WINDOW_SIZE, start_seq=7)
        db = _mock_db(recent, count=SUMMARY_THRESHOLD + 1)
        service = ConversationContextService(db=db)

        result = await service.get_context(_SESSION_ID)

        assert len(result) == SLIDING_WINDOW_SIZE

    @pytest.mark.asyncio
    async def test_fifty_messages_returns_sliding_window(self) -> None:
        """50 messages still returns only SLIDING_WINDOW_SIZE recent."""
        recent = _make_messages(SLIDING_WINDOW_SIZE, start_seq=36)
        db = _mock_db(recent, count=50)
        service = ConversationContextService(db=db)

        result = await service.get_context(_SESSION_ID)

        assert len(result) == SLIDING_WINDOW_SIZE


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


class TestBuildSummary:
    """Tests for build_summary method."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_older_messages(self) -> None:
        """No summary needed if there are no messages before the sequence."""
        db = _mock_db([], count=0)
        generator = AsyncMock(return_value="summary")
        service = ConversationContextService(db=db, summary_generator=generator)

        result = await service.build_summary(_SESSION_ID, up_to_sequence=1)

        assert result is None
        generator.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returns_none_when_no_generator(self) -> None:
        """No summary when generator is not configured."""
        older = _make_messages(5)
        db = _mock_db(older, count=5)
        service = ConversationContextService(db=db, summary_generator=None)

        result = await service.build_summary(_SESSION_ID, up_to_sequence=6)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_summary_text(self) -> None:
        """Summary text returned when generator succeeds."""
        older = _make_messages(5)
        db = _mock_db(older, count=5)
        generator = AsyncMock(return_value="Topics: combat rules, movement")
        service = ConversationContextService(db=db, summary_generator=generator)

        result = await service.build_summary(_SESSION_ID, up_to_sequence=6)

        assert result == "Topics: combat rules, movement"
        generator.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_summary_generator_receives_correct_format(self) -> None:
        """Generator receives system prompt + user/assistant messages."""
        older = [
            _make_message(1, msg_type=MessageType.USER_QUESTION, content="How does combat work?"),
            _make_message(2, msg_type=MessageType.AI_ANSWER, content="Combat uses d20 rolls."),
        ]
        db = _mock_db(older, count=2)
        generator = AsyncMock(return_value="summary")
        service = ConversationContextService(db=db, summary_generator=generator)

        await service.build_summary(_SESSION_ID, up_to_sequence=3)

        call_kwargs = generator.call_args
        chat_messages = call_kwargs.kwargs["messages"]
        assert chat_messages[0]["role"] == "system"
        assert chat_messages[1]["role"] == "user"
        assert chat_messages[1]["content"] == "How does combat work?"
        assert chat_messages[2]["role"] == "assistant"
        assert chat_messages[2]["content"] == "Combat uses d20 rolls."

    @pytest.mark.asyncio
    async def test_summary_failure_returns_none(self) -> None:
        """Summary generation failure returns None (graceful degradation)."""
        older = _make_messages(5)
        db = _mock_db(older, count=5)
        generator = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        service = ConversationContextService(db=db, summary_generator=generator)

        result = await service.build_summary(_SESSION_ID, up_to_sequence=6)

        assert result is None


# ---------------------------------------------------------------------------
# get_context_with_summary — integration of window + summary
# ---------------------------------------------------------------------------


class TestGetContextWithSummary:
    """Tests for the combined get_context_with_summary method."""

    @pytest.mark.asyncio
    async def test_long_conversation_with_summary(self) -> None:
        """Long conversation returns sliding window and summary text."""
        recent = _make_messages(SLIDING_WINDOW_SIZE, start_seq=7)
        older = _make_messages(6, start_seq=1)
        all_msgs = older + recent

        db = AsyncMock()
        call_tracker = {"n": 0}

        async def execute_side_effect(stmt: Any) -> MagicMock:
            call_tracker["n"] += 1
            stmt_str = str(stmt)
            result = MagicMock()
            if "count" in stmt_str.lower():
                result.scalar_one.return_value = len(all_msgs)
            else:
                # Return recent for window queries, older for summary queries
                # Heuristic: if limit is involved, it's a window query
                result.scalars.return_value.all.return_value = recent
            return result

        db.execute = AsyncMock(side_effect=execute_side_effect)

        generator = AsyncMock(return_value="Game discussed: combat rules")
        service = ConversationContextService(db=db, summary_generator=generator)

        messages, summary = await service.get_context_with_summary(_SESSION_ID)

        assert len(messages) == SLIDING_WINDOW_SIZE
        assert summary == "Game discussed: combat rules"

    @pytest.mark.asyncio
    async def test_summary_failure_expands_window(self) -> None:
        """When summary fails, window expands to DEGRADED_WINDOW_SIZE."""
        recent_small = _make_messages(SLIDING_WINDOW_SIZE, start_seq=16)
        recent_large = _make_messages(DEGRADED_WINDOW_SIZE, start_seq=1)

        db = AsyncMock()
        fetch_call = {"n": 0}

        async def execute_side_effect(stmt: Any) -> MagicMock:
            stmt_str = str(stmt)
            result = MagicMock()
            if "count" in stmt_str.lower():
                result.scalar_one.return_value = 50
            else:
                fetch_call["n"] += 1
                # 1st fetch: sliding window (15), 2nd: older for summary,
                # 3rd: degraded window (30)
                if fetch_call["n"] == 1:
                    result.scalars.return_value.all.return_value = recent_small
                elif fetch_call["n"] == 2:
                    # Older messages for summary attempt
                    result.scalars.return_value.all.return_value = _make_messages(5)
                else:
                    result.scalars.return_value.all.return_value = recent_large
            return result

        db.execute = AsyncMock(side_effect=execute_side_effect)

        generator = AsyncMock(side_effect=RuntimeError("LLM down"))
        service = ConversationContextService(db=db, summary_generator=generator)

        messages, summary = await service.get_context_with_summary(_SESSION_ID)

        assert len(messages) == DEGRADED_WINDOW_SIZE
        assert summary is None

    @pytest.mark.asyncio
    async def test_no_generator_long_conversation(self) -> None:
        """Long conversation without generator returns window, no summary, no expansion."""
        recent = _make_messages(SLIDING_WINDOW_SIZE, start_seq=6)

        db = AsyncMock()

        async def execute_side_effect(stmt: Any) -> MagicMock:
            stmt_str = str(stmt)
            result = MagicMock()
            if "count" in stmt_str.lower():
                result.scalar_one.return_value = 30
            else:
                result.scalars.return_value.all.return_value = recent
            return result

        db.execute = AsyncMock(side_effect=execute_side_effect)

        service = ConversationContextService(db=db, summary_generator=None)

        messages, summary = await service.get_context_with_summary(_SESSION_ID)

        # No generator configured, so no degradation either
        assert len(messages) == SLIDING_WINDOW_SIZE
        assert summary is None


# ---------------------------------------------------------------------------
# Message ordering
# ---------------------------------------------------------------------------


class TestMessageOrdering:
    """Tests that messages are returned in correct sequence order."""

    @pytest.mark.asyncio
    async def test_messages_ordered_by_sequence_ascending(self) -> None:
        """Returned messages are ordered by sequence ascending."""
        msgs = _make_messages(5)
        db = _mock_db(msgs, count=5)
        service = ConversationContextService(db=db)

        result = await service.get_context(_SESSION_ID)

        sequences = [m.sequence for m in result]
        assert sequences == sorted(sequences)


# ---------------------------------------------------------------------------
# Context budget constants
# ---------------------------------------------------------------------------


class TestContextBudgetConstants:
    """Verify context budget allocation constants sum to 1.0."""

    def test_budgets_sum_to_one(self) -> None:
        """KG + conversation + ad-hoc budgets total 100%."""
        from tabletop_oracle.services.ai.conversation import (
            CONTEXT_BUDGET_AD_HOC,
            CONTEXT_BUDGET_CONVERSATION,
            CONTEXT_BUDGET_KG,
        )

        total = CONTEXT_BUDGET_KG + CONTEXT_BUDGET_CONVERSATION + CONTEXT_BUDGET_AD_HOC
        assert total == pytest.approx(1.0)

    def test_kg_budget_is_largest(self) -> None:
        """KG results get the largest context budget share."""
        from tabletop_oracle.services.ai.conversation import (
            CONTEXT_BUDGET_AD_HOC,
            CONTEXT_BUDGET_CONVERSATION,
            CONTEXT_BUDGET_KG,
        )

        assert CONTEXT_BUDGET_KG > CONTEXT_BUDGET_CONVERSATION > CONTEXT_BUDGET_AD_HOC


# ---------------------------------------------------------------------------
# Custom max_context_messages
# ---------------------------------------------------------------------------


class TestCustomThreshold:
    """Tests for non-default max_context_messages parameter."""

    @pytest.mark.asyncio
    async def test_custom_lower_threshold(self) -> None:
        """Custom threshold of 10 triggers sliding window at 11 messages."""
        recent = _make_messages(SLIDING_WINDOW_SIZE, start_seq=1)
        db = _mock_db(recent, count=11)
        service = ConversationContextService(db=db)

        result = await service.get_context(_SESSION_ID, max_context_messages=10)

        assert len(result) == SLIDING_WINDOW_SIZE

    @pytest.mark.asyncio
    async def test_custom_higher_threshold(self) -> None:
        """Custom threshold of 50 returns all 30 messages."""
        msgs = _make_messages(30)
        db = _mock_db(msgs, count=30)
        service = ConversationContextService(db=db)

        result = await service.get_context(_SESSION_ID, max_context_messages=50)

        assert len(result) == 30
