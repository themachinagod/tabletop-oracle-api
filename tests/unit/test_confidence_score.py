"""Unit tests for ConfidenceScoreStage.

Tests:
- Confidence score between 0 and 1 (AC-612)
- Label assignment: High (0.8-1.0), Moderate (0.5-0.79), Low (0.0-0.49)
- Uses intent_analysis model slot (fast, cheap)
- Justification logged (not shown to user)
- stream.confidence SSE event emitted
- Token usage and model call count tracked on context
- Score clamping for out-of-range LLM responses
- Graceful handling of malformed LLM responses
- Prompt includes question, answer, and source summary
"""

from __future__ import annotations

import json
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tabletop_oracle.models.enums import ModelCapability
from tabletop_oracle.services.ai.stages.confidence_score import (
    ConfidenceScoreStage,
    _clamp_score,
    _parse_response,
    _score_to_label,
)
from tabletop_oracle.services.model.types import CompletionResult
from tabletop_oracle.streaming.event_types import StreamEventType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SESSION_ID = uuid.uuid4()
_MESSAGE_ID = uuid.uuid4()
_GAME_ID = uuid.uuid4()
_USER_ID = uuid.uuid4()


def _make_ctx(
    *,
    question: str = "How does flanking work?",
    answer_text: str = "Flanking gives advantage on melee attacks.",
    retrieval_results: list[Any] | None = None,
    traversal_results: list[Any] | None = None,
    token_usage: int = 0,
    model_call_count: int = 0,
) -> MagicMock:
    """Build a mock PipelineContext for confidence scoring.

    Args:
        question: The player's question text.
        answer_text: The generated answer text.
        retrieval_results: Mock retrieval results from KG search.
        traversal_results: Mock traversal results from KG traversal.
        token_usage: Starting token usage count.
        model_call_count: Starting model call count.

    Returns:
        MagicMock mimicking a PipelineContext dataclass.
    """
    ctx = MagicMock()
    ctx.session.id = _SESSION_ID
    ctx.session.user_id = _USER_ID
    ctx.user_message.id = _MESSAGE_ID
    ctx.user_message.content = question
    ctx.game_id = _GAME_ID
    ctx.answer_text = answer_text
    ctx.retrieval_results = retrieval_results or []
    ctx.traversal_results = traversal_results or []
    ctx.confidence_score = None
    ctx.confidence_label = None
    ctx.token_usage = token_usage
    ctx.model_call_count = model_call_count
    return ctx


def _make_retrieval_result(
    content: str = "Some rule content",
    score: float = 0.85,
) -> MagicMock:
    """Build a mock RetrievalResult.

    Args:
        content: The content text.
        score: Similarity score.

    Returns:
        MagicMock mimicking a RetrievalResult.
    """
    result = MagicMock()
    result.content = content
    result.score = score
    return result


def _make_traversal_result(
    content: str = "Related rule content",
    depth: int = 1,
) -> MagicMock:
    """Build a mock TraversalResult.

    Args:
        content: The content text.
        depth: Traversal depth.

    Returns:
        MagicMock mimicking a TraversalResult.
    """
    result = MagicMock()
    result.content = content
    result.depth = depth
    return result


def _make_completion_result(
    response: dict[str, Any],
    input_tokens: int = 80,
    output_tokens: int = 30,
) -> CompletionResult:
    """Build a CompletionResult with a JSON response body.

    Args:
        response: Dict to serialize as the LLM response content.
        input_tokens: Input token count.
        output_tokens: Output token count.

    Returns:
        CompletionResult with JSON content.
    """
    return CompletionResult(
        content=json.dumps(response),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model_used="openai/gpt-4o-mini",
        is_fallback=False,
    )


def _make_model_client(
    response: dict[str, Any] | None = None,
    input_tokens: int = 80,
    output_tokens: int = 30,
) -> AsyncMock:
    """Build a mocked ModelClient.

    Args:
        response: Dict for the JSON response. Defaults to high confidence.
        input_tokens: Input token count on the result.
        output_tokens: Output token count on the result.

    Returns:
        AsyncMock mimicking ModelClient.
    """
    if response is None:
        response = {
            "score": 0.92,
            "justification": "Answer is directly supported by 3 agreeing sources.",
        }
    client = AsyncMock()
    client.complete.return_value = _make_completion_result(
        response,
        input_tokens,
        output_tokens,
    )
    return client


async def _collect_events(stage: ConfidenceScoreStage, ctx: MagicMock, client: AsyncMock) -> list:
    """Run the stage and collect all yielded SSE events.

    Args:
        stage: The stage instance.
        ctx: Mock pipeline context.
        client: Mock model client.

    Returns:
        List of SseEvent instances.
    """
    return [event async for event in stage.execute(ctx, client)]


# ---------------------------------------------------------------------------
# Tests: Confidence Score Range (AC-612)
# ---------------------------------------------------------------------------


class TestConfidenceScoreRange:
    """Tests that confidence score is between 0 and 1."""

    async def test_score_stored_on_context(self) -> None:
        """Score from LLM is written to ctx.confidence_score."""
        ctx = _make_ctx()
        client = _make_model_client({"score": 0.75, "justification": "Partial support."})
        stage = ConfidenceScoreStage()

        await _collect_events(stage, ctx, client)

        assert ctx.confidence_score == 0.75

    async def test_high_score_stored(self) -> None:
        """Score of 1.0 is valid and stored."""
        ctx = _make_ctx()
        client = _make_model_client({"score": 1.0, "justification": "Perfect."})
        stage = ConfidenceScoreStage()

        await _collect_events(stage, ctx, client)

        assert ctx.confidence_score == 1.0

    async def test_zero_score_stored(self) -> None:
        """Score of 0.0 is valid and stored."""
        ctx = _make_ctx()
        client = _make_model_client({"score": 0.0, "justification": "No support."})
        stage = ConfidenceScoreStage()

        await _collect_events(stage, ctx, client)

        assert ctx.confidence_score == 0.0

    async def test_score_above_one_clamped(self) -> None:
        """Score above 1.0 is clamped to 1.0."""
        ctx = _make_ctx()
        client = _make_model_client({"score": 1.5, "justification": "Overconfident."})
        stage = ConfidenceScoreStage()

        await _collect_events(stage, ctx, client)

        assert ctx.confidence_score == 1.0

    async def test_score_below_zero_clamped(self) -> None:
        """Score below 0.0 is clamped to 0.0."""
        ctx = _make_ctx()
        client = _make_model_client({"score": -0.3, "justification": "Negative."})
        stage = ConfidenceScoreStage()

        await _collect_events(stage, ctx, client)

        assert ctx.confidence_score == 0.0


# ---------------------------------------------------------------------------
# Tests: Label Assignment
# ---------------------------------------------------------------------------


class TestLabelAssignment:
    """Tests for confidence label assignment based on score thresholds."""

    async def test_high_label_at_threshold(self) -> None:
        """Score of exactly 0.8 gets 'High' label."""
        ctx = _make_ctx()
        client = _make_model_client({"score": 0.8, "justification": "Good."})
        stage = ConfidenceScoreStage()

        await _collect_events(stage, ctx, client)

        assert ctx.confidence_label == "High"

    async def test_high_label_above_threshold(self) -> None:
        """Score of 0.95 gets 'High' label."""
        ctx = _make_ctx()
        client = _make_model_client({"score": 0.95, "justification": "Great."})
        stage = ConfidenceScoreStage()

        await _collect_events(stage, ctx, client)

        assert ctx.confidence_label == "High"

    async def test_moderate_label_at_threshold(self) -> None:
        """Score of exactly 0.5 gets 'Moderate' label."""
        ctx = _make_ctx()
        client = _make_model_client({"score": 0.5, "justification": "Okay."})
        stage = ConfidenceScoreStage()

        await _collect_events(stage, ctx, client)

        assert ctx.confidence_label == "Moderate"

    async def test_moderate_label_mid_range(self) -> None:
        """Score of 0.65 gets 'Moderate' label."""
        ctx = _make_ctx()
        client = _make_model_client({"score": 0.65, "justification": "Partial."})
        stage = ConfidenceScoreStage()

        await _collect_events(stage, ctx, client)

        assert ctx.confidence_label == "Moderate"

    async def test_moderate_label_just_below_high(self) -> None:
        """Score of 0.79 gets 'Moderate' label."""
        ctx = _make_ctx()
        client = _make_model_client({"score": 0.79, "justification": "Close."})
        stage = ConfidenceScoreStage()

        await _collect_events(stage, ctx, client)

        assert ctx.confidence_label == "Moderate"

    async def test_low_label_below_threshold(self) -> None:
        """Score of 0.3 gets 'Low' label."""
        ctx = _make_ctx()
        client = _make_model_client({"score": 0.3, "justification": "Weak."})
        stage = ConfidenceScoreStage()

        await _collect_events(stage, ctx, client)

        assert ctx.confidence_label == "Low"

    async def test_low_label_at_zero(self) -> None:
        """Score of 0.0 gets 'Low' label."""
        ctx = _make_ctx()
        client = _make_model_client({"score": 0.0, "justification": "None."})
        stage = ConfidenceScoreStage()

        await _collect_events(stage, ctx, client)

        assert ctx.confidence_label == "Low"

    async def test_low_label_just_below_moderate(self) -> None:
        """Score of 0.49 gets 'Low' label."""
        ctx = _make_ctx()
        client = _make_model_client({"score": 0.49, "justification": "Weak."})
        stage = ConfidenceScoreStage()

        await _collect_events(stage, ctx, client)

        assert ctx.confidence_label == "Low"

    def test_score_to_label_boundaries(self) -> None:
        """Verify label boundaries directly via the helper function."""
        assert _score_to_label(1.0) == "High"
        assert _score_to_label(0.8) == "High"
        assert _score_to_label(0.79) == "Moderate"
        assert _score_to_label(0.5) == "Moderate"
        assert _score_to_label(0.49) == "Low"
        assert _score_to_label(0.0) == "Low"


# ---------------------------------------------------------------------------
# Tests: Model Slot Usage
# ---------------------------------------------------------------------------


class TestModelSlotUsage:
    """Tests that the intent_analysis model slot is used (fast, cheap)."""

    async def test_uses_intent_analysis_capability(self) -> None:
        """ModelClient.complete called with INTENT_ANALYSIS capability."""
        ctx = _make_ctx()
        client = _make_model_client()
        stage = ConfidenceScoreStage()

        await _collect_events(stage, ctx, client)

        client.complete.assert_awaited_once()
        call_kwargs = client.complete.call_args
        assert call_kwargs.kwargs["capability"] == ModelCapability.INTENT_ANALYSIS

    async def test_attribution_has_session_and_message(self) -> None:
        """Token attribution includes session and message IDs."""
        ctx = _make_ctx()
        client = _make_model_client()
        stage = ConfidenceScoreStage()

        await _collect_events(stage, ctx, client)

        call_kwargs = client.complete.call_args
        attribution = call_kwargs.kwargs["attribution"]
        assert attribution.session_id == _SESSION_ID
        assert attribution.message_id == _MESSAGE_ID


# ---------------------------------------------------------------------------
# Tests: Token Usage Tracking
# ---------------------------------------------------------------------------


class TestTokenTracking:
    """Tests for token usage and model call count tracking."""

    async def test_token_usage_incremented(self) -> None:
        """Token usage on context is incremented by input + output tokens."""
        ctx = _make_ctx(token_usage=500)
        client = _make_model_client(input_tokens=80, output_tokens=30)
        stage = ConfidenceScoreStage()

        await _collect_events(stage, ctx, client)

        assert ctx.token_usage == 500 + 80 + 30

    async def test_model_call_count_incremented(self) -> None:
        """Model call count on context is incremented by 1."""
        ctx = _make_ctx(model_call_count=5)
        client = _make_model_client()
        stage = ConfidenceScoreStage()

        await _collect_events(stage, ctx, client)

        assert ctx.model_call_count == 6


# ---------------------------------------------------------------------------
# Tests: SSE Event Emission
# ---------------------------------------------------------------------------


class TestSseEventEmission:
    """Tests for stream.confidence SSE event emission."""

    async def test_emits_confidence_event(self) -> None:
        """Stage yields exactly one stream.confidence event."""
        ctx = _make_ctx()
        client = _make_model_client({"score": 0.85, "justification": "Good."})
        stage = ConfidenceScoreStage()

        events = await _collect_events(stage, ctx, client)

        assert len(events) == 1
        assert events[0].type == StreamEventType.CONFIDENCE

    async def test_event_payload_contains_score(self) -> None:
        """Event payload includes the confidence score."""
        ctx = _make_ctx()
        client = _make_model_client({"score": 0.72, "justification": "Okay."})
        stage = ConfidenceScoreStage()

        events = await _collect_events(stage, ctx, client)

        assert events[0].payload["score"] == 0.72

    async def test_event_payload_contains_label(self) -> None:
        """Event payload includes the confidence label."""
        ctx = _make_ctx()
        client = _make_model_client({"score": 0.72, "justification": "Okay."})
        stage = ConfidenceScoreStage()

        events = await _collect_events(stage, ctx, client)

        assert events[0].payload["label"] == "Moderate"

    async def test_event_payload_does_not_contain_justification(self) -> None:
        """Justification is logged but NOT in the SSE event payload."""
        ctx = _make_ctx()
        client = _make_model_client({"score": 0.9, "justification": "Should not appear in event."})
        stage = ConfidenceScoreStage()

        events = await _collect_events(stage, ctx, client)

        assert "justification" not in events[0].payload


# ---------------------------------------------------------------------------
# Tests: JSON Response Parsing
# ---------------------------------------------------------------------------


class TestResponseParsing:
    """Tests for structured JSON response parsing."""

    def test_valid_json_parsed_correctly(self) -> None:
        """Valid JSON with all fields is parsed without error."""
        content = json.dumps({"score": 0.85, "justification": "Good support."})
        result = _parse_response(content)
        assert result["score"] == 0.85
        assert result["justification"] == "Good support."

    def test_markdown_code_fences_stripped(self) -> None:
        """JSON wrapped in markdown code fences is parsed correctly."""
        content = '```json\n{"score": 0.7, "justification": "Partial."}\n```'
        result = _parse_response(content)
        assert result["score"] == 0.7

    def test_plain_code_fences_stripped(self) -> None:
        """JSON wrapped in plain code fences is parsed."""
        content = '```\n{"score": 0.6, "justification": "Okay."}\n```'
        result = _parse_response(content)
        assert result["score"] == 0.6

    def test_invalid_json_raises_value_error(self) -> None:
        """Non-JSON content raises ValueError."""
        with pytest.raises(ValueError, match="not valid JSON"):
            _parse_response("This is not JSON at all")

    def test_non_object_json_raises_value_error(self) -> None:
        """JSON array raises ValueError."""
        with pytest.raises(ValueError, match="not a JSON object"):
            _parse_response("[0.5, 0.6]")

    def test_missing_score_key_raises_value_error(self) -> None:
        """Missing 'score' key raises ValueError."""
        content = json.dumps({"justification": "No score."})
        with pytest.raises(ValueError, match="missing required keys"):
            _parse_response(content)

    def test_missing_justification_key_raises_value_error(self) -> None:
        """Missing 'justification' key raises ValueError."""
        content = json.dumps({"score": 0.5})
        with pytest.raises(ValueError, match="missing required keys"):
            _parse_response(content)

    def test_non_numeric_score_raises_value_error(self) -> None:
        """Non-numeric score raises ValueError."""
        content = json.dumps({"score": "high", "justification": "Invalid."})
        with pytest.raises(ValueError, match="must be a number"):
            _parse_response(content)

    def test_integer_score_accepted(self) -> None:
        """Integer score (e.g. 1) is accepted as valid."""
        content = json.dumps({"score": 1, "justification": "Perfect."})
        result = _parse_response(content)
        assert result["score"] == 1

    def test_extra_keys_accepted(self) -> None:
        """Extra keys in the response are accepted (forward-compatible)."""
        content = json.dumps({"score": 0.8, "justification": "Good.", "extra": "ignored"})
        result = _parse_response(content)
        assert result["score"] == 0.8


# ---------------------------------------------------------------------------
# Tests: Score Clamping
# ---------------------------------------------------------------------------


class TestScoreClamping:
    """Tests for defensive score clamping."""

    def test_clamp_above_one(self) -> None:
        """Score above 1.0 is clamped to 1.0."""
        assert _clamp_score(1.5) == 1.0

    def test_clamp_below_zero(self) -> None:
        """Score below 0.0 is clamped to 0.0."""
        assert _clamp_score(-0.1) == 0.0

    def test_clamp_normal_value(self) -> None:
        """Score in valid range is unchanged."""
        assert _clamp_score(0.75) == 0.75

    def test_clamp_boundary_one(self) -> None:
        """Score of exactly 1.0 is unchanged."""
        assert _clamp_score(1.0) == 1.0

    def test_clamp_boundary_zero(self) -> None:
        """Score of exactly 0.0 is unchanged."""
        assert _clamp_score(0.0) == 0.0


# ---------------------------------------------------------------------------
# Tests: Prompt Construction
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    """Tests that the prompt includes required context."""

    async def test_prompt_includes_question(self) -> None:
        """The user's question appears in the prompt messages."""
        ctx = _make_ctx(question="What is opportunity attack?")
        client = _make_model_client()
        stage = ConfidenceScoreStage()

        await _collect_events(stage, ctx, client)

        messages = client.complete.call_args.kwargs["messages"]
        user_msg = messages[1]["content"]
        assert "What is opportunity attack?" in user_msg

    async def test_prompt_includes_answer(self) -> None:
        """The generated answer appears in the prompt messages."""
        ctx = _make_ctx(answer_text="An opportunity attack is a reaction.")
        client = _make_model_client()
        stage = ConfidenceScoreStage()

        await _collect_events(stage, ctx, client)

        messages = client.complete.call_args.kwargs["messages"]
        user_msg = messages[1]["content"]
        assert "An opportunity attack is a reaction." in user_msg

    async def test_prompt_includes_source_count(self) -> None:
        """Source count appears in the prompt."""
        results = [_make_retrieval_result(), _make_retrieval_result()]
        ctx = _make_ctx(retrieval_results=results)
        client = _make_model_client()
        stage = ConfidenceScoreStage()

        await _collect_events(stage, ctx, client)

        messages = client.complete.call_args.kwargs["messages"]
        user_msg = messages[1]["content"]
        assert "2 total" in user_msg

    async def test_prompt_includes_retrieval_content(self) -> None:
        """Retrieval result content appears in the prompt."""
        results = [_make_retrieval_result(content="Flanking rule text", score=0.9)]
        ctx = _make_ctx(retrieval_results=results)
        client = _make_model_client()
        stage = ConfidenceScoreStage()

        await _collect_events(stage, ctx, client)

        messages = client.complete.call_args.kwargs["messages"]
        user_msg = messages[1]["content"]
        assert "Flanking rule text" in user_msg
        assert "score=0.90" in user_msg

    async def test_prompt_includes_traversal_content(self) -> None:
        """Traversal result content appears in the prompt."""
        results = [_make_traversal_result(content="Related mechanic", depth=2)]
        ctx = _make_ctx(traversal_results=results)
        client = _make_model_client()
        stage = ConfidenceScoreStage()

        await _collect_events(stage, ctx, client)

        messages = client.complete.call_args.kwargs["messages"]
        user_msg = messages[1]["content"]
        assert "Related mechanic" in user_msg
        assert "depth=2" in user_msg

    async def test_prompt_no_sources_shows_fallback(self) -> None:
        """When no sources exist, the prompt shows 'No sources available'."""
        ctx = _make_ctx(retrieval_results=[], traversal_results=[])
        client = _make_model_client()
        stage = ConfidenceScoreStage()

        await _collect_events(stage, ctx, client)

        messages = client.complete.call_args.kwargs["messages"]
        user_msg = messages[1]["content"]
        assert "No sources available" in user_msg

    async def test_prompt_has_system_and_user_messages(self) -> None:
        """Prompt has exactly two messages: system and user."""
        ctx = _make_ctx()
        client = _make_model_client()
        stage = ConfidenceScoreStage()

        await _collect_events(stage, ctx, client)

        messages = client.complete.call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


# ---------------------------------------------------------------------------
# Tests: Error Propagation
# ---------------------------------------------------------------------------


class TestErrorPropagation:
    """Tests that LLM and parsing errors propagate correctly."""

    async def test_malformed_json_raises_value_error(self) -> None:
        """Malformed JSON from LLM raises ValueError through the stage."""
        ctx = _make_ctx()
        client = AsyncMock()
        client.complete.return_value = CompletionResult(
            content="Not JSON",
            input_tokens=50,
            output_tokens=10,
            model_used="test",
            is_fallback=False,
        )
        stage = ConfidenceScoreStage()

        with pytest.raises(ValueError, match="not valid JSON"):
            await _collect_events(stage, ctx, client)

    async def test_model_client_error_propagates(self) -> None:
        """Exceptions from ModelClient.complete propagate unchanged."""
        ctx = _make_ctx()
        client = AsyncMock()
        client.complete.side_effect = RuntimeError("model exploded")
        stage = ConfidenceScoreStage()

        with pytest.raises(RuntimeError, match="model exploded"):
            await _collect_events(stage, ctx, client)
