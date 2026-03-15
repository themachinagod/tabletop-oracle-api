"""Unit tests for IntentAnalysisStage.

Tests:
- Intent extracted as free text (AC-601)
- Ambiguity detected with reason (AC-603)
- Structured JSON response parsed correctly
- Uses intent_analysis model slot (AC-701)
- Token usage and model call count tracked on context
- Graceful handling of malformed LLM responses
- Prompt includes question, conversation history, session config, ad-hoc context
"""

from __future__ import annotations

import json
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tabletop_oracle.models.enums import ModelCapability
from tabletop_oracle.services.ai.stages.intent_analysis import (
    IntentAnalysisStage,
    _parse_response,
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
    question: str = "How does flanking work?",
    conversation_history: list[Any] | None = None,
    ad_hoc_text_context: str | None = None,
    ad_hoc_image_descriptions: list[str] | None = None,
    player_count: int | None = 4,
    game_name: str = "Dungeons & Dragons",
    token_usage: int = 0,
    model_call_count: int = 0,
) -> MagicMock:
    """Build a mock PipelineContext for intent analysis.

    Args:
        question: The player's question text.
        conversation_history: Prior messages for context.
        ad_hoc_text_context: Player-provided text context.
        ad_hoc_image_descriptions: Processed image descriptions.
        player_count: Number of players in the session.
        game_name: Name of the game being played.
        token_usage: Starting token usage count.
        model_call_count: Starting model call count.

    Returns:
        MagicMock mimicking a PipelineContext dataclass.
    """
    ctx = MagicMock()
    ctx.session.id = _SESSION_ID
    ctx.session.user_id = _USER_ID
    ctx.session.game.name = game_name
    ctx.user_message.id = _MESSAGE_ID
    ctx.user_message.content = question
    ctx.game_id = _GAME_ID
    ctx.player_count = player_count
    ctx.conversation_history = conversation_history or []
    ctx.ad_hoc_text_context = ad_hoc_text_context
    ctx.ad_hoc_image_descriptions = ad_hoc_image_descriptions or []
    ctx.intent = None
    ctx.is_ambiguous = False
    ctx.token_usage = token_usage
    ctx.model_call_count = model_call_count
    return ctx


def _make_completion_result(
    response: dict[str, Any],
    input_tokens: int = 100,
    output_tokens: int = 50,
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
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> AsyncMock:
    """Build a mocked ModelClient.

    Args:
        response: Dict for the JSON response. Defaults to a non-ambiguous intent.
        input_tokens: Input token count on the result.
        output_tokens: Output token count on the result.

    Returns:
        AsyncMock mimicking ModelClient.
    """
    if response is None:
        response = {
            "intent": "Rules question about flanking mechanics",
            "is_ambiguous": False,
            "ambiguity_reason": None,
            "suggested_clarification": None,
        }
    client = AsyncMock()
    client.complete.return_value = _make_completion_result(
        response,
        input_tokens,
        output_tokens,
    )
    return client


# ---------------------------------------------------------------------------
# Tests: Intent Extraction (AC-601)
# ---------------------------------------------------------------------------


class TestIntentExtraction:
    """Tests for intent extraction as free text."""

    async def test_intent_extracted_to_context(self) -> None:
        """Intent string from LLM is written to ctx.intent."""
        response = {
            "intent": "Question about movement rules in combat",
            "is_ambiguous": False,
            "ambiguity_reason": None,
            "suggested_clarification": None,
        }
        ctx = _make_ctx()
        client = _make_model_client(response)
        stage = IntentAnalysisStage()

        async for _ in stage.execute(ctx, client):
            pass

        assert ctx.intent == "Question about movement rules in combat"

    async def test_empty_intent_string_accepted(self) -> None:
        """An empty intent string is accepted without error."""
        response = {
            "intent": "",
            "is_ambiguous": False,
            "ambiguity_reason": None,
            "suggested_clarification": None,
        }
        ctx = _make_ctx()
        client = _make_model_client(response)
        stage = IntentAnalysisStage()

        async for _ in stage.execute(ctx, client):
            pass

        assert ctx.intent == ""


# ---------------------------------------------------------------------------
# Tests: Ambiguity Detection (AC-603)
# ---------------------------------------------------------------------------


class TestAmbiguityDetection:
    """Tests for ambiguity detection with reason."""

    async def test_ambiguous_question_flagged(self) -> None:
        """Ambiguous question sets is_ambiguous to True on context."""
        response = {
            "intent": "Question about attack mechanics",
            "is_ambiguous": True,
            "ambiguity_reason": "Could refer to melee or ranged attacks",
            "suggested_clarification": "Are you asking about melee or ranged attacks?",
        }
        ctx = _make_ctx(question="How do attacks work?")
        client = _make_model_client(response)
        stage = IntentAnalysisStage()

        async for _ in stage.execute(ctx, client):
            pass

        assert ctx.is_ambiguous is True

    async def test_non_ambiguous_question_not_flagged(self) -> None:
        """Clear question sets is_ambiguous to False on context."""
        response = {
            "intent": "Flanking rules in melee combat",
            "is_ambiguous": False,
            "ambiguity_reason": None,
            "suggested_clarification": None,
        }
        ctx = _make_ctx()
        client = _make_model_client(response)
        stage = IntentAnalysisStage()

        async for _ in stage.execute(ctx, client):
            pass

        assert ctx.is_ambiguous is False


# ---------------------------------------------------------------------------
# Tests: Model Slot Usage (AC-701)
# ---------------------------------------------------------------------------


class TestModelSlotUsage:
    """Tests that the intent_analysis model slot is used."""

    async def test_uses_intent_analysis_capability(self) -> None:
        """ModelClient.complete called with INTENT_ANALYSIS capability."""
        ctx = _make_ctx()
        client = _make_model_client()
        stage = IntentAnalysisStage()

        async for _ in stage.execute(ctx, client):
            pass

        client.complete.assert_awaited_once()
        call_kwargs = client.complete.call_args
        assert call_kwargs.kwargs["capability"] == ModelCapability.INTENT_ANALYSIS

    async def test_attribution_has_session_and_message(self) -> None:
        """Token attribution includes session and message IDs."""
        ctx = _make_ctx()
        client = _make_model_client()
        stage = IntentAnalysisStage()

        async for _ in stage.execute(ctx, client):
            pass

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
        ctx = _make_ctx(token_usage=200)
        client = _make_model_client(input_tokens=150, output_tokens=75)
        stage = IntentAnalysisStage()

        async for _ in stage.execute(ctx, client):
            pass

        assert ctx.token_usage == 200 + 150 + 75

    async def test_model_call_count_incremented(self) -> None:
        """Model call count on context is incremented by 1."""
        ctx = _make_ctx(model_call_count=2)
        client = _make_model_client()
        stage = IntentAnalysisStage()

        async for _ in stage.execute(ctx, client):
            pass

        assert ctx.model_call_count == 3


# ---------------------------------------------------------------------------
# Tests: JSON Response Parsing
# ---------------------------------------------------------------------------


class TestResponseParsing:
    """Tests for structured JSON response parsing."""

    def test_valid_json_parsed_correctly(self) -> None:
        """Valid JSON with all fields is parsed without error."""
        content = json.dumps(
            {
                "intent": "Something",
                "is_ambiguous": True,
                "ambiguity_reason": "Reason",
                "suggested_clarification": "Clarify?",
            }
        )
        result = _parse_response(content)
        assert result["intent"] == "Something"
        assert result["is_ambiguous"] is True

    def test_markdown_code_fences_stripped(self) -> None:
        """JSON wrapped in markdown code fences is parsed correctly."""
        content = '```json\n{"intent": "test", "is_ambiguous": false}\n```'
        result = _parse_response(content)
        assert result["intent"] == "test"

    def test_plain_code_fences_stripped(self) -> None:
        """JSON wrapped in plain code fences (no language tag) is parsed."""
        content = '```\n{"intent": "test", "is_ambiguous": false}\n```'
        result = _parse_response(content)
        assert result["intent"] == "test"

    def test_invalid_json_raises_value_error(self) -> None:
        """Non-JSON content raises ValueError."""
        with pytest.raises(ValueError, match="not valid JSON"):
            _parse_response("This is not JSON at all")

    def test_non_object_json_raises_value_error(self) -> None:
        """JSON array raises ValueError."""
        with pytest.raises(ValueError, match="not a JSON object"):
            _parse_response('["list", "not", "object"]')

    def test_missing_intent_key_raises_value_error(self) -> None:
        """Missing 'intent' key raises ValueError."""
        content = json.dumps({"is_ambiguous": False})
        with pytest.raises(ValueError, match="missing required keys"):
            _parse_response(content)

    def test_missing_is_ambiguous_key_raises_value_error(self) -> None:
        """Missing 'is_ambiguous' key raises ValueError."""
        content = json.dumps({"intent": "something"})
        with pytest.raises(ValueError, match="missing required keys"):
            _parse_response(content)

    def test_extra_keys_accepted(self) -> None:
        """Extra keys in the response are accepted (forward-compatible)."""
        content = json.dumps(
            {
                "intent": "test",
                "is_ambiguous": False,
                "extra_field": "ignored",
            }
        )
        result = _parse_response(content)
        assert result["intent"] == "test"


# ---------------------------------------------------------------------------
# Tests: Prompt Construction
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    """Tests that the prompt includes required context."""

    async def test_prompt_includes_question(self) -> None:
        """The user's question appears in the prompt messages."""
        ctx = _make_ctx(question="What is opportunity attack?")
        client = _make_model_client()
        stage = IntentAnalysisStage()

        async for _ in stage.execute(ctx, client):
            pass

        messages = client.complete.call_args.kwargs["messages"]
        user_msg = messages[1]["content"]
        assert "What is opportunity attack?" in user_msg

    async def test_prompt_includes_game_name(self) -> None:
        """The game name appears in the prompt messages."""
        ctx = _make_ctx(game_name="Gloomhaven")
        client = _make_model_client()
        stage = IntentAnalysisStage()

        async for _ in stage.execute(ctx, client):
            pass

        messages = client.complete.call_args.kwargs["messages"]
        user_msg = messages[1]["content"]
        assert "Gloomhaven" in user_msg

    async def test_prompt_includes_player_count(self) -> None:
        """Player count appears in the prompt when set."""
        ctx = _make_ctx(player_count=6)
        client = _make_model_client()
        stage = IntentAnalysisStage()

        async for _ in stage.execute(ctx, client):
            pass

        messages = client.complete.call_args.kwargs["messages"]
        user_msg = messages[1]["content"]
        assert "Player count: 6" in user_msg

    async def test_prompt_omits_player_count_when_none(self) -> None:
        """Player count line is omitted when player_count is None."""
        ctx = _make_ctx(player_count=None)
        client = _make_model_client()
        stage = IntentAnalysisStage()

        async for _ in stage.execute(ctx, client):
            pass

        messages = client.complete.call_args.kwargs["messages"]
        user_msg = messages[1]["content"]
        assert "Player count:" not in user_msg

    async def test_prompt_includes_conversation_history(self) -> None:
        """Conversation history messages appear in the prompt."""
        msg1 = MagicMock()
        msg1.type = "user_question"
        msg1.content = "What is AC?"
        msg2 = MagicMock()
        msg2.type = "ai_answer"
        msg2.content = "AC stands for Armor Class."

        ctx = _make_ctx(conversation_history=[msg1, msg2])
        client = _make_model_client()
        stage = IntentAnalysisStage()

        async for _ in stage.execute(ctx, client):
            pass

        messages = client.complete.call_args.kwargs["messages"]
        user_msg = messages[1]["content"]
        assert "What is AC?" in user_msg
        assert "AC stands for Armor Class." in user_msg
        assert "Conversation history:" in user_msg

    async def test_prompt_includes_ad_hoc_text_context(self) -> None:
        """Ad-hoc text context appears in the prompt."""
        ctx = _make_ctx(ad_hoc_text_context="The rogue has advantage on this attack")
        client = _make_model_client()
        stage = IntentAnalysisStage()

        async for _ in stage.execute(ctx, client):
            pass

        messages = client.complete.call_args.kwargs["messages"]
        user_msg = messages[1]["content"]
        assert "The rogue has advantage on this attack" in user_msg

    async def test_prompt_includes_image_descriptions(self) -> None:
        """Image descriptions appear in the prompt."""
        ctx = _make_ctx(ad_hoc_image_descriptions=["A game board with tokens"])
        client = _make_model_client()
        stage = IntentAnalysisStage()

        async for _ in stage.execute(ctx, client):
            pass

        messages = client.complete.call_args.kwargs["messages"]
        user_msg = messages[1]["content"]
        assert "A game board with tokens" in user_msg

    async def test_prompt_has_system_and_user_messages(self) -> None:
        """Prompt has exactly two messages: system and user."""
        ctx = _make_ctx()
        client = _make_model_client()
        stage = IntentAnalysisStage()

        async for _ in stage.execute(ctx, client):
            pass

        messages = client.complete.call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


# ---------------------------------------------------------------------------
# Tests: No Events Yielded
# ---------------------------------------------------------------------------


class TestNoEventsYielded:
    """Tests that the stage yields no SSE events."""

    async def test_yields_no_events(self) -> None:
        """Stage produces no SSE events (silent stage)."""
        ctx = _make_ctx()
        client = _make_model_client()
        stage = IntentAnalysisStage()
        events = [event async for event in stage.execute(ctx, client)]
        assert events == []


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
        stage = IntentAnalysisStage()

        with pytest.raises(ValueError, match="not valid JSON"):
            async for _ in stage.execute(ctx, client):
                pass

    async def test_model_client_error_propagates(self) -> None:
        """Exceptions from ModelClient.complete propagate unchanged."""
        ctx = _make_ctx()
        client = AsyncMock()
        client.complete.side_effect = RuntimeError("model exploded")
        stage = IntentAnalysisStage()

        with pytest.raises(RuntimeError, match="model exploded"):
            async for _ in stage.execute(ctx, client):
                pass


# ---------------------------------------------------------------------------
# Tests: Game Name Fallback
# ---------------------------------------------------------------------------


class TestGameNameFallback:
    """Tests for game name resolution fallback."""

    async def test_missing_game_relationship_uses_fallback(self) -> None:
        """When session.game is None, falls back to 'Unknown game'."""
        ctx = _make_ctx()
        ctx.session.game = None
        client = _make_model_client()
        stage = IntentAnalysisStage()

        async for _ in stage.execute(ctx, client):
            pass

        messages = client.complete.call_args.kwargs["messages"]
        user_msg = messages[1]["content"]
        assert "Unknown game" in user_msg
