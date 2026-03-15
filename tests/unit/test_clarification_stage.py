"""Unit tests for ClarificationStage.

Tests:
- Non-ambiguous questions pass through (no-op) (AC-603)
- Clarification message persisted as ai_clarification type (AC-603)
- Round counting correct from thread (max 2)
- After 2 rounds, best-effort answer produced (is_ambiguous cleared)
- Pipeline stop signal set when clarification persisted
- Token usage tracked on LLM call
- Fallback to suggested_clarification on LLM failure
- Generic fallback when no suggestion available
- Clarification prompt includes question and ambiguity reason
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

from tabletop_oracle.models.enums import MessageType, ModelCapability
from tabletop_oracle.services.ai.stages.clarification import (
    MAX_CLARIFICATION_ROUNDS,
    ClarificationStage,
    _count_clarification_rounds,
)
from tabletop_oracle.services.model.types import CompletionResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SESSION_ID = uuid.uuid4()
_MESSAGE_ID = uuid.uuid4()
_GAME_ID = uuid.uuid4()
_USER_ID = uuid.uuid4()


def _make_ctx(
    *,
    question: str = "How do attacks work?",
    is_ambiguous: bool = True,
    ambiguity_reason: str | None = "Could refer to melee or ranged",
    suggested_clarification: str | None = "Are you asking about melee or ranged attacks?",
    conversation_history: list[MagicMock] | None = None,
    token_usage: int = 0,
    model_call_count: int = 0,
    message_sequence: int = 5,
    game_name: str = "Dungeons & Dragons",
) -> MagicMock:
    """Build a mock PipelineContext for clarification tests.

    Args:
        question: The player's question text.
        is_ambiguous: Whether intent analysis flagged ambiguity.
        ambiguity_reason: Why the question is ambiguous.
        suggested_clarification: Draft clarification from intent analysis.
        conversation_history: Prior messages for thread counting.
        token_usage: Starting token count.
        model_call_count: Starting model call count.
        message_sequence: Sequence number of the user message.
        game_name: Name of the game.

    Returns:
        MagicMock mimicking PipelineContext.
    """
    ctx = MagicMock()
    ctx.session.id = _SESSION_ID
    ctx.session.user_id = _USER_ID
    ctx.session.game.name = game_name
    ctx.user_message.id = _MESSAGE_ID
    ctx.user_message.content = question
    ctx.user_message.sequence = message_sequence
    ctx.game_id = _GAME_ID
    ctx.is_ambiguous = is_ambiguous
    ctx.ambiguity_reason = ambiguity_reason
    ctx.suggested_clarification = suggested_clarification
    ctx.clarification_round = 0
    ctx.conversation_history = conversation_history or []
    ctx.token_usage = token_usage
    ctx.model_call_count = model_call_count
    ctx.intent = "Question about attack mechanics"
    ctx.player_count = 4
    ctx.ad_hoc_text_context = None
    ctx.ad_hoc_image_descriptions = []
    return ctx


def _make_clarification_message() -> MagicMock:
    """Build a mock Message of type ai_clarification."""
    msg = MagicMock()
    msg.type = MessageType.AI_CLARIFICATION
    msg.content = "Are you asking about melee or ranged?"
    return msg


def _make_user_message() -> MagicMock:
    """Build a mock Message of type user_question."""
    msg = MagicMock()
    msg.type = MessageType.USER_QUESTION
    msg.content = "How do attacks work?"
    return msg


def _make_answer_message() -> MagicMock:
    """Build a mock Message of type ai_answer."""
    msg = MagicMock()
    msg.type = MessageType.AI_ANSWER
    msg.content = "Attacks use a d20 roll."
    return msg


def _make_model_client(
    response_text: str = "Could you clarify whether you mean melee or ranged attacks?",
    input_tokens: int = 80,
    output_tokens: int = 30,
) -> AsyncMock:
    """Build a mocked ModelClient.

    Args:
        response_text: Text the model returns.
        input_tokens: Input token count on the result.
        output_tokens: Output token count on the result.

    Returns:
        AsyncMock mimicking ModelClient.
    """
    client = AsyncMock()
    client.complete.return_value = CompletionResult(
        content=response_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model_used="openai/gpt-4o-mini",
        is_fallback=False,
    )
    return client


def _make_failing_model_client() -> AsyncMock:
    """Build a mocked ModelClient that raises on complete."""
    client = AsyncMock()
    client.complete.side_effect = RuntimeError("model unavailable")
    return client


# ---------------------------------------------------------------------------
# Tests: Non-Ambiguous Pass-Through
# ---------------------------------------------------------------------------


class TestNonAmbiguousPassThrough:
    """Tests that non-ambiguous questions pass through without action."""

    async def test_not_ambiguous_is_noop(self) -> None:
        """When is_ambiguous is False, stage does nothing."""
        ctx = _make_ctx(is_ambiguous=False)
        db = AsyncMock()
        client = _make_model_client()
        stage = ClarificationStage(db=db)

        events = [event async for event in stage.execute(ctx, client)]

        assert events == []
        assert stage.pipeline_should_stop is False
        client.complete.assert_not_awaited()
        db.add.assert_not_called()

    async def test_not_ambiguous_preserves_context(self) -> None:
        """Non-ambiguous pass-through does not modify token counters."""
        ctx = _make_ctx(is_ambiguous=False, token_usage=100, model_call_count=2)
        db = AsyncMock()
        client = _make_model_client()
        stage = ClarificationStage(db=db)

        async for _ in stage.execute(ctx, client):
            pass

        assert ctx.token_usage == 100
        assert ctx.model_call_count == 2


# ---------------------------------------------------------------------------
# Tests: Round Counting
# ---------------------------------------------------------------------------


class TestRoundCounting:
    """Tests for clarification round counting from conversation thread."""

    def test_zero_rounds_with_no_history(self) -> None:
        """No conversation history means zero rounds."""
        ctx = _make_ctx(conversation_history=[])
        assert _count_clarification_rounds(ctx) == 0

    def test_zero_rounds_with_no_clarifications(self) -> None:
        """History with only user and answer messages means zero rounds."""
        ctx = _make_ctx(
            conversation_history=[_make_user_message(), _make_answer_message()],
        )
        assert _count_clarification_rounds(ctx) == 0

    def test_one_round_with_one_clarification(self) -> None:
        """One ai_clarification message in history counts as one round."""
        ctx = _make_ctx(
            conversation_history=[
                _make_user_message(),
                _make_clarification_message(),
                _make_user_message(),
            ],
        )
        assert _count_clarification_rounds(ctx) == 1

    def test_two_rounds_with_two_clarifications(self) -> None:
        """Two ai_clarification messages count as two rounds."""
        ctx = _make_ctx(
            conversation_history=[
                _make_user_message(),
                _make_clarification_message(),
                _make_user_message(),
                _make_clarification_message(),
                _make_user_message(),
            ],
        )
        assert _count_clarification_rounds(ctx) == 2

    def test_mixed_message_types_counted_correctly(self) -> None:
        """Only ai_clarification messages are counted, not answers or system."""
        system_msg = MagicMock()
        system_msg.type = MessageType.SYSTEM
        system_msg.content = "Session started"

        ctx = _make_ctx(
            conversation_history=[
                system_msg,
                _make_user_message(),
                _make_answer_message(),
                _make_clarification_message(),
                _make_user_message(),
            ],
        )
        assert _count_clarification_rounds(ctx) == 1


# ---------------------------------------------------------------------------
# Tests: Max Rounds Best-Effort (AC-603)
# ---------------------------------------------------------------------------


class TestMaxRoundsBestEffort:
    """Tests that after max rounds, pipeline proceeds with best-effort."""

    async def test_max_rounds_clears_ambiguity(self) -> None:
        """After MAX_CLARIFICATION_ROUNDS, is_ambiguous is set to False."""
        history = [
            _make_user_message(),
            _make_clarification_message(),
            _make_user_message(),
            _make_clarification_message(),
            _make_user_message(),
        ]
        ctx = _make_ctx(conversation_history=history)
        db = AsyncMock()
        client = _make_model_client()
        stage = ClarificationStage(db=db)

        async for _ in stage.execute(ctx, client):
            pass

        assert ctx.is_ambiguous is False
        assert ctx.clarification_round == 2
        assert stage.pipeline_should_stop is False
        client.complete.assert_not_awaited()

    async def test_max_rounds_does_not_persist_message(self) -> None:
        """No clarification message persisted when max rounds reached."""
        history = [
            _make_user_message(),
            _make_clarification_message(),
            _make_user_message(),
            _make_clarification_message(),
            _make_user_message(),
        ]
        ctx = _make_ctx(conversation_history=history)
        db = AsyncMock()
        client = _make_model_client()
        stage = ClarificationStage(db=db)

        async for _ in stage.execute(ctx, client):
            pass

        db.add.assert_not_called()

    async def test_exceeding_max_rounds_still_clears_ambiguity(self) -> None:
        """Even with >2 clarifications in history, ambiguity is cleared."""
        history = [
            _make_user_message(),
            _make_clarification_message(),
            _make_user_message(),
            _make_clarification_message(),
            _make_user_message(),
            _make_clarification_message(),
            _make_user_message(),
        ]
        ctx = _make_ctx(conversation_history=history)
        db = AsyncMock()
        client = _make_model_client()
        stage = ClarificationStage(db=db)

        async for _ in stage.execute(ctx, client):
            pass

        assert ctx.is_ambiguous is False


# ---------------------------------------------------------------------------
# Tests: Clarification Message Persistence (AC-603)
# ---------------------------------------------------------------------------


class TestClarificationPersistence:
    """Tests for clarification message creation and persistence."""

    async def test_clarification_message_added_to_db(self) -> None:
        """Clarification message is added to the database session."""
        ctx = _make_ctx(conversation_history=[])
        db = AsyncMock()
        client = _make_model_client()
        stage = ClarificationStage(db=db)

        async for _ in stage.execute(ctx, client):
            pass

        db.add.assert_called_once()
        db.flush.assert_awaited_once()

    async def test_clarification_message_type(self) -> None:
        """Persisted message has type ai_clarification."""
        ctx = _make_ctx(conversation_history=[])
        db = AsyncMock()
        client = _make_model_client()
        stage = ClarificationStage(db=db)

        async for _ in stage.execute(ctx, client):
            pass

        message = db.add.call_args[0][0]
        assert message.type == MessageType.AI_CLARIFICATION

    async def test_clarification_message_in_reply_to(self) -> None:
        """Persisted message has in_reply_to_id pointing to user question."""
        ctx = _make_ctx(conversation_history=[])
        db = AsyncMock()
        client = _make_model_client()
        stage = ClarificationStage(db=db)

        async for _ in stage.execute(ctx, client):
            pass

        message = db.add.call_args[0][0]
        assert message.in_reply_to_id == _MESSAGE_ID

    async def test_clarification_message_session_id(self) -> None:
        """Persisted message belongs to the correct session."""
        ctx = _make_ctx(conversation_history=[])
        db = AsyncMock()
        client = _make_model_client()
        stage = ClarificationStage(db=db)

        async for _ in stage.execute(ctx, client):
            pass

        message = db.add.call_args[0][0]
        assert message.session_id == _SESSION_ID

    async def test_clarification_message_sequence(self) -> None:
        """Persisted message sequence is user message sequence + 1."""
        ctx = _make_ctx(conversation_history=[], message_sequence=10)
        db = AsyncMock()
        client = _make_model_client()
        stage = ClarificationStage(db=db)

        async for _ in stage.execute(ctx, client):
            pass

        message = db.add.call_args[0][0]
        assert message.sequence == 11

    async def test_clarification_message_content_from_llm(self) -> None:
        """Persisted message content comes from the LLM response."""
        expected_text = "Do you mean melee attacks or ranged attacks?"
        ctx = _make_ctx(conversation_history=[])
        db = AsyncMock()
        client = _make_model_client(response_text=f"  {expected_text}  ")
        stage = ClarificationStage(db=db)

        async for _ in stage.execute(ctx, client):
            pass

        message = db.add.call_args[0][0]
        assert message.content == expected_text


# ---------------------------------------------------------------------------
# Tests: Pipeline Stop Signal
# ---------------------------------------------------------------------------


class TestPipelineStopSignal:
    """Tests for the pipeline_should_stop flag."""

    async def test_stop_signal_set_on_clarification(self) -> None:
        """pipeline_should_stop is True after persisting a clarification."""
        ctx = _make_ctx(conversation_history=[])
        db = AsyncMock()
        client = _make_model_client()
        stage = ClarificationStage(db=db)

        async for _ in stage.execute(ctx, client):
            pass

        assert stage.pipeline_should_stop is True

    async def test_stop_signal_not_set_on_noop(self) -> None:
        """pipeline_should_stop is False when question is not ambiguous."""
        ctx = _make_ctx(is_ambiguous=False)
        db = AsyncMock()
        client = _make_model_client()
        stage = ClarificationStage(db=db)

        async for _ in stage.execute(ctx, client):
            pass

        assert stage.pipeline_should_stop is False

    async def test_stop_signal_not_set_on_max_rounds(self) -> None:
        """pipeline_should_stop is False when max rounds reached."""
        history = [
            _make_user_message(),
            _make_clarification_message(),
            _make_user_message(),
            _make_clarification_message(),
            _make_user_message(),
        ]
        ctx = _make_ctx(conversation_history=history)
        db = AsyncMock()
        client = _make_model_client()
        stage = ClarificationStage(db=db)

        async for _ in stage.execute(ctx, client):
            pass

        assert stage.pipeline_should_stop is False

    def test_reset_clears_stop_signal(self) -> None:
        """reset() sets pipeline_should_stop back to False."""
        stage = ClarificationStage(db=AsyncMock())
        stage.pipeline_should_stop = True
        stage.reset()
        assert stage.pipeline_should_stop is False


# ---------------------------------------------------------------------------
# Tests: Token Usage Tracking
# ---------------------------------------------------------------------------


class TestTokenTracking:
    """Tests for token usage and model call count tracking."""

    async def test_token_usage_incremented(self) -> None:
        """Token usage on context is incremented by LLM call tokens."""
        ctx = _make_ctx(conversation_history=[], token_usage=200)
        db = AsyncMock()
        client = _make_model_client(input_tokens=80, output_tokens=30)
        stage = ClarificationStage(db=db)

        async for _ in stage.execute(ctx, client):
            pass

        assert ctx.token_usage == 200 + 80 + 30

    async def test_model_call_count_incremented(self) -> None:
        """Model call count on context is incremented by 1."""
        ctx = _make_ctx(conversation_history=[], model_call_count=1)
        db = AsyncMock()
        client = _make_model_client()
        stage = ClarificationStage(db=db)

        async for _ in stage.execute(ctx, client):
            pass

        assert ctx.model_call_count == 2


# ---------------------------------------------------------------------------
# Tests: Model Slot Usage
# ---------------------------------------------------------------------------


class TestModelSlotUsage:
    """Tests that the clarification_generation model slot is used."""

    async def test_uses_clarification_generation_capability(self) -> None:
        """ModelClient.complete called with CLARIFICATION_GENERATION capability."""
        ctx = _make_ctx(conversation_history=[])
        db = AsyncMock()
        client = _make_model_client()
        stage = ClarificationStage(db=db)

        async for _ in stage.execute(ctx, client):
            pass

        client.complete.assert_awaited_once()
        call_kwargs = client.complete.call_args
        assert call_kwargs.kwargs["capability"] == ModelCapability.CLARIFICATION_GENERATION

    async def test_attribution_has_session_and_message(self) -> None:
        """Token attribution includes session and message IDs."""
        ctx = _make_ctx(conversation_history=[])
        db = AsyncMock()
        client = _make_model_client()
        stage = ClarificationStage(db=db)

        async for _ in stage.execute(ctx, client):
            pass

        call_kwargs = client.complete.call_args
        attribution = call_kwargs.kwargs["attribution"]
        assert attribution.session_id == _SESSION_ID
        assert attribution.message_id == _MESSAGE_ID


# ---------------------------------------------------------------------------
# Tests: Fallback on LLM Failure
# ---------------------------------------------------------------------------


class TestFallbackBehavior:
    """Tests for graceful fallback when LLM call fails."""

    async def test_fallback_to_suggested_clarification(self) -> None:
        """When LLM fails, uses suggested_clarification from intent analysis."""
        suggestion = "Are you asking about melee or ranged attacks?"
        ctx = _make_ctx(
            conversation_history=[],
            suggested_clarification=suggestion,
        )
        db = AsyncMock()
        client = _make_failing_model_client()
        stage = ClarificationStage(db=db)

        async for _ in stage.execute(ctx, client):
            pass

        message = db.add.call_args[0][0]
        assert message.content == suggestion
        assert stage.pipeline_should_stop is True

    async def test_fallback_to_generic_when_no_suggestion(self) -> None:
        """When LLM fails and no suggestion, uses generic fallback."""
        ctx = _make_ctx(
            conversation_history=[],
            suggested_clarification=None,
        )
        db = AsyncMock()
        client = _make_failing_model_client()
        stage = ClarificationStage(db=db)

        async for _ in stage.execute(ctx, client):
            pass

        message = db.add.call_args[0][0]
        assert "clarify" in message.content.lower()
        assert stage.pipeline_should_stop is True

    async def test_fallback_does_not_increment_token_usage(self) -> None:
        """Token usage not incremented when LLM call fails."""
        ctx = _make_ctx(
            conversation_history=[],
            token_usage=100,
            model_call_count=1,
        )
        db = AsyncMock()
        client = _make_failing_model_client()
        stage = ClarificationStage(db=db)

        async for _ in stage.execute(ctx, client):
            pass

        assert ctx.token_usage == 100
        assert ctx.model_call_count == 1


# ---------------------------------------------------------------------------
# Tests: No Events Yielded
# ---------------------------------------------------------------------------


class TestNoEventsYielded:
    """Tests that the stage yields no SSE events in any path."""

    async def test_yields_no_events_on_clarification(self) -> None:
        """Stage produces no SSE events when generating clarification."""
        ctx = _make_ctx(conversation_history=[])
        db = AsyncMock()
        client = _make_model_client()
        stage = ClarificationStage(db=db)
        events = [event async for event in stage.execute(ctx, client)]
        assert events == []

    async def test_yields_no_events_on_noop(self) -> None:
        """Stage produces no SSE events on non-ambiguous pass-through."""
        ctx = _make_ctx(is_ambiguous=False)
        db = AsyncMock()
        client = _make_model_client()
        stage = ClarificationStage(db=db)
        events = [event async for event in stage.execute(ctx, client)]
        assert events == []

    async def test_yields_no_events_on_max_rounds(self) -> None:
        """Stage produces no SSE events when max rounds reached."""
        history = [
            _make_user_message(),
            _make_clarification_message(),
            _make_user_message(),
            _make_clarification_message(),
            _make_user_message(),
        ]
        ctx = _make_ctx(conversation_history=history)
        db = AsyncMock()
        client = _make_model_client()
        stage = ClarificationStage(db=db)
        events = [event async for event in stage.execute(ctx, client)]
        assert events == []


# ---------------------------------------------------------------------------
# Tests: Clarification Round Set on Context
# ---------------------------------------------------------------------------


class TestClarificationRoundOnContext:
    """Tests that clarification_round is set correctly on context."""

    async def test_round_set_to_zero_on_first_clarification(self) -> None:
        """First clarification sets clarification_round to 0."""
        ctx = _make_ctx(conversation_history=[])
        db = AsyncMock()
        client = _make_model_client()
        stage = ClarificationStage(db=db)

        async for _ in stage.execute(ctx, client):
            pass

        assert ctx.clarification_round == 0

    async def test_round_set_to_one_on_second_clarification(self) -> None:
        """Second clarification (1 prior) sets clarification_round to 1."""
        history = [
            _make_user_message(),
            _make_clarification_message(),
            _make_user_message(),
        ]
        ctx = _make_ctx(conversation_history=history)
        db = AsyncMock()
        client = _make_model_client()
        stage = ClarificationStage(db=db)

        async for _ in stage.execute(ctx, client):
            pass

        assert ctx.clarification_round == 1

    async def test_round_set_to_two_on_max(self) -> None:
        """Max rounds sets clarification_round to 2."""
        history = [
            _make_user_message(),
            _make_clarification_message(),
            _make_user_message(),
            _make_clarification_message(),
            _make_user_message(),
        ]
        ctx = _make_ctx(conversation_history=history)
        db = AsyncMock()
        client = _make_model_client()
        stage = ClarificationStage(db=db)

        async for _ in stage.execute(ctx, client):
            pass

        assert ctx.clarification_round == 2


# ---------------------------------------------------------------------------
# Tests: MAX_CLARIFICATION_ROUNDS constant
# ---------------------------------------------------------------------------


class TestMaxRoundsConstant:
    """Tests that the max rounds constant is set correctly."""

    def test_max_rounds_is_two(self) -> None:
        """MAX_CLARIFICATION_ROUNDS is 2 per PRD-006 Section 3.2."""
        assert MAX_CLARIFICATION_ROUNDS == 2
