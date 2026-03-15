"""Unit tests for AnswerSynthesisStage.

Tests:
- Answer grounded in KG content via prompt constraints (AC-601, AC-604)
- Source markers embedded for citation mapping (AC-602)
- Strategy content framed as guidance in prompt (AC-607)
- Source conflicts noted with priority in prompt (AC-609)
- Streaming yields stream.token SSE events (AC-610)
- Insufficient knowledge acknowledged via prompt (AC-604)
- Answer text accumulated in ctx.answer_text
- Uses answer_synthesis model slot
- Token attribution with session and message
- Model call count incremented
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tabletop_oracle.models.enums import ModelCapability
from tabletop_oracle.services.ai.context import RetrievalResult, TraversalResult
from tabletop_oracle.services.ai.prompts.answer_synthesis import (
    build_answer_synthesis_messages,
)
from tabletop_oracle.services.ai.stages.answer_synthesis import AnswerSynthesisStage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SESSION_ID = uuid.uuid4()
_MESSAGE_ID = uuid.uuid4()
_GAME_ID = uuid.uuid4()
_USER_ID = uuid.uuid4()
_CONCEPT_ID_1 = uuid.uuid4()
_CONCEPT_ID_2 = uuid.uuid4()
_CONCEPT_ID_3 = uuid.uuid4()


def _make_retrieval_result(
    *,
    concept_id: uuid.UUID | None = None,
    content: str = "When flanking, attackers get advantage.",
    score: float = 0.95,
    document_name: str = "Core Rulebook",
    document_type: str = "core_rules",
    section_path: str = "Chapter 9: Combat",
    page_number: int | None = 42,
    is_authoritative: bool = True,
) -> RetrievalResult:
    """Build a RetrievalResult for testing.

    Args:
        concept_id: KG concept ID. Defaults to a random UUID.
        content: The concept text content.
        score: Similarity score.
        document_name: Source document name.
        document_type: Source document type.
        section_path: Section path within the document.
        page_number: Page number if available.
        is_authoritative: Whether the source is authoritative.

    Returns:
        A RetrievalResult for use in pipeline context.
    """
    return RetrievalResult(
        concept_id=concept_id or uuid.uuid4(),
        content=content,
        score=score,
        source={
            "document_name": document_name,
            "document_type": document_type,
            "section_path": section_path,
            "page_number": page_number,
            "is_authoritative": is_authoritative,
        },
    )


def _make_traversal_result(
    *,
    concept_id: uuid.UUID | None = None,
    content: str = "Opportunity attacks",
    relationship: str = "related_to",
    depth: int = 1,
) -> TraversalResult:
    """Build a TraversalResult for testing.

    Args:
        concept_id: KG concept ID. Defaults to a random UUID.
        content: The concept text content.
        relationship: Relationship type.
        depth: Traversal depth.

    Returns:
        A TraversalResult for use in pipeline context.
    """
    return TraversalResult(
        concept_id=concept_id or uuid.uuid4(),
        content=content,
        relationship=relationship,
        depth=depth,
        source={"concept_type": "rule"},
    )


def _make_ctx(
    *,
    question: str = "How does flanking work?",
    retrieval_results: list[RetrievalResult] | None = None,
    traversal_results: list[TraversalResult] | None = None,
    knowledge_insufficient: bool = False,
    conversation_history: list[Any] | None = None,
    ad_hoc_text_context: str | None = None,
    ad_hoc_image_descriptions: list[str] | None = None,
    player_count: int | None = 4,
    game_name: str = "Dungeons & Dragons",
    token_usage: int = 0,
    model_call_count: int = 0,
) -> MagicMock:
    """Build a mock PipelineContext for answer synthesis.

    Args:
        question: The player's question text.
        retrieval_results: KG retrieval results.
        traversal_results: KG traversal results.
        knowledge_insufficient: Whether retrieval found nothing.
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
    if retrieval_results is not None:
        ctx.retrieval_results = retrieval_results
    else:
        ctx.retrieval_results = [
            _make_retrieval_result(concept_id=_CONCEPT_ID_1),
        ]
    ctx.traversal_results = traversal_results or []
    ctx.knowledge_insufficient = knowledge_insufficient
    ctx.intent = "Rules question about flanking mechanics"
    ctx.answer_text = ""
    ctx.token_usage = token_usage
    ctx.model_call_count = model_call_count
    return ctx


def _make_streaming_model_client(
    tokens: list[str] | None = None,
) -> AsyncMock:
    """Build a mocked ModelClient with streaming behaviour.

    Args:
        tokens: List of token strings to yield. Defaults to a short answer.

    Returns:
        AsyncMock mimicking ModelClient with a working stream() method.
    """
    if tokens is None:
        tokens = [
            "Flanking ",
            "gives ",
            "advantage ",
            "[1].",
        ]

    async def mock_stream(
        capability: Any,
        messages: Any,
        attribution: Any,
        cancel_check: Any = None,
    ) -> Any:
        """Async generator yielding tokens."""
        for token in tokens:
            yield token

    client = AsyncMock()
    client.stream = mock_stream  # type: ignore[assignment]
    return client


# ---------------------------------------------------------------------------
# Tests: Streaming Token Events (AC-610)
# ---------------------------------------------------------------------------


class TestStreamingTokenEvents:
    """Tests that streaming yields stream.token SSE events."""

    async def test_yields_stream_token_events(self) -> None:
        """Each token from ModelClient.stream() becomes a stream.token event."""
        tokens = ["Hello ", "world ", "[1]."]
        ctx = _make_ctx()
        client = _make_streaming_model_client(tokens)
        stage = AnswerSynthesisStage()

        events = [event async for event in stage.execute(ctx, client)]

        assert len(events) == len(tokens)
        for event, expected_token in zip(events, tokens, strict=True):
            assert event.type == "stream.token"
            assert event.payload["token"] == expected_token

    async def test_empty_stream_yields_no_events(self) -> None:
        """An empty token stream produces no events."""
        ctx = _make_ctx()
        client = _make_streaming_model_client(tokens=[])
        stage = AnswerSynthesisStage()

        events = [event async for event in stage.execute(ctx, client)]

        assert events == []

    async def test_single_token_stream(self) -> None:
        """A single-token stream yields exactly one event."""
        ctx = _make_ctx()
        client = _make_streaming_model_client(tokens=["answer"])
        stage = AnswerSynthesisStage()

        events = [event async for event in stage.execute(ctx, client)]

        assert len(events) == 1
        assert events[0].payload["token"] == "answer"


# ---------------------------------------------------------------------------
# Tests: Answer Text Accumulation
# ---------------------------------------------------------------------------


class TestAnswerAccumulation:
    """Tests that answer text is accumulated in ctx.answer_text."""

    async def test_answer_text_accumulated(self) -> None:
        """All tokens are joined and stored in ctx.answer_text."""
        tokens = ["Flanking ", "gives ", "advantage ", "[1]."]
        ctx = _make_ctx()
        client = _make_streaming_model_client(tokens)
        stage = AnswerSynthesisStage()

        async for _ in stage.execute(ctx, client):
            pass

        assert ctx.answer_text == "Flanking gives advantage [1]."

    async def test_empty_stream_sets_empty_answer(self) -> None:
        """An empty token stream sets answer_text to empty string."""
        ctx = _make_ctx()
        client = _make_streaming_model_client(tokens=[])
        stage = AnswerSynthesisStage()

        async for _ in stage.execute(ctx, client):
            pass

        assert ctx.answer_text == ""


# ---------------------------------------------------------------------------
# Tests: Model Slot Usage
# ---------------------------------------------------------------------------


class TestModelSlotUsage:
    """Tests that the answer_synthesis model slot is used."""

    async def test_uses_answer_synthesis_capability(self) -> None:
        """ModelClient.stream called with ANSWER_SYNTHESIS capability."""
        ctx = _make_ctx()

        captured_capability = None

        async def capturing_stream(
            capability: Any, messages: Any, attribution: Any, **kwargs: Any
        ) -> Any:
            nonlocal captured_capability
            captured_capability = capability
            return
            yield  # pragma: no cover

        client = AsyncMock()
        client.stream = capturing_stream  # type: ignore[assignment]
        stage = AnswerSynthesisStage()

        async for _ in stage.execute(ctx, client):
            pass

        assert captured_capability == ModelCapability.ANSWER_SYNTHESIS

    async def test_attribution_has_session_and_message(self) -> None:
        """Token attribution includes session and message IDs."""
        ctx = _make_ctx()

        captured_attribution = None

        async def capturing_stream(
            capability: Any, messages: Any, attribution: Any, **kwargs: Any
        ) -> Any:
            nonlocal captured_attribution
            captured_attribution = attribution
            return
            yield  # pragma: no cover

        client = AsyncMock()
        client.stream = capturing_stream  # type: ignore[assignment]
        stage = AnswerSynthesisStage()

        async for _ in stage.execute(ctx, client):
            pass

        assert captured_attribution is not None
        assert captured_attribution.session_id == _SESSION_ID
        assert captured_attribution.message_id == _MESSAGE_ID
        assert captured_attribution.user_id == _USER_ID


# ---------------------------------------------------------------------------
# Tests: Model Call Count
# ---------------------------------------------------------------------------


class TestModelCallCount:
    """Tests that model_call_count is incremented."""

    async def test_model_call_count_incremented(self) -> None:
        """Model call count on context is incremented by 1."""
        ctx = _make_ctx(model_call_count=2)
        client = _make_streaming_model_client()
        stage = AnswerSynthesisStage()

        async for _ in stage.execute(ctx, client):
            pass

        assert ctx.model_call_count == 3


# ---------------------------------------------------------------------------
# Tests: Prompt Grounding Constraints (AC-601, AC-604)
# ---------------------------------------------------------------------------


class TestGroundingConstraints:
    """Tests that the system prompt enforces grounding rules."""

    def test_system_prompt_forbids_fabrication(self) -> None:
        """System prompt instructs no fabrication."""
        ctx = _make_ctx()
        messages = build_answer_synthesis_messages(ctx)
        system_msg = messages[0]["content"]

        assert "Do not fabricate" in system_msg

    def test_system_prompt_requires_grounding(self) -> None:
        """System prompt requires answers based only on provided knowledge."""
        ctx = _make_ctx()
        messages = build_answer_synthesis_messages(ctx)
        system_msg = messages[0]["content"]

        assert "ONLY" in system_msg
        assert "knowledge provided" in system_msg


# ---------------------------------------------------------------------------
# Tests: Source Markers for Citation Mapping (AC-602)
# ---------------------------------------------------------------------------


class TestSourceMarkers:
    """Tests that the prompt includes numbered source references."""

    def test_knowledge_section_has_numbered_markers(self) -> None:
        """Knowledge section includes [1], [2] markers for sources."""
        results = [
            _make_retrieval_result(content="Rule A", document_name="Book A"),
            _make_retrieval_result(content="Rule B", document_name="Book B"),
        ]
        ctx = _make_ctx(retrieval_results=results)
        messages = build_answer_synthesis_messages(ctx)
        user_msg = messages[1]["content"]

        assert "[1]" in user_msg
        assert "[2]" in user_msg

    def test_source_markers_instruction_in_system_prompt(self) -> None:
        """System prompt instructs the model to include source markers."""
        ctx = _make_ctx()
        messages = build_answer_synthesis_messages(ctx)
        system_msg = messages[0]["content"]

        assert "[1]" in system_msg
        assert "[2]" in system_msg
        assert "source marker" in system_msg.lower()


# ---------------------------------------------------------------------------
# Tests: Strategy Content Framing (AC-607)
# ---------------------------------------------------------------------------


class TestStrategyFraming:
    """Tests that strategy content is framed as guidance."""

    def test_system_prompt_frames_strategy_as_guidance(self) -> None:
        """System prompt instructs strategy content to be framed as guidance."""
        ctx = _make_ctx()
        messages = build_answer_synthesis_messages(ctx)
        system_msg = messages[0]["content"]

        assert "guidance" in system_msg.lower() or "suggestion" in system_msg.lower()
        assert "strategy" in system_msg.lower()

    def test_strategy_document_type_visible_in_knowledge(self) -> None:
        """Strategy document type is visible in the knowledge section."""
        results = [
            _make_retrieval_result(
                content="Use this opening strategy",
                document_type="strategy",
                document_name="Strategy Guide",
            ),
        ]
        ctx = _make_ctx(retrieval_results=results)
        messages = build_answer_synthesis_messages(ctx)
        user_msg = messages[1]["content"]

        assert "strategy" in user_msg.lower()


# ---------------------------------------------------------------------------
# Tests: Source Priority / Conflict Resolution (AC-609)
# ---------------------------------------------------------------------------


class TestSourcePriority:
    """Tests that the prompt establishes document type priority."""

    def test_system_prompt_defines_priority_order(self) -> None:
        """System prompt defines Errata > FAQ > Core Rules > Other."""
        ctx = _make_ctx()
        messages = build_answer_synthesis_messages(ctx)
        system_msg = messages[0]["content"]

        assert "Errata" in system_msg
        assert "FAQ" in system_msg
        assert "Core Rules" in system_msg

        errata_pos = system_msg.index("Errata")
        faq_pos = system_msg.index("FAQ")
        core_pos = system_msg.index("Core Rules")
        assert errata_pos < faq_pos < core_pos

    def test_system_prompt_instructs_conflict_reporting(self) -> None:
        """System prompt instructs noting conflicts with priority."""
        ctx = _make_ctx()
        messages = build_answer_synthesis_messages(ctx)
        system_msg = messages[0]["content"]

        assert "conflict" in system_msg.lower()
        assert "priority" in system_msg.lower()

    def test_document_types_shown_in_knowledge_section(self) -> None:
        """Document type labels are included in knowledge entries."""
        results = [
            _make_retrieval_result(content="Original rule", document_type="core_rules"),
            _make_retrieval_result(content="Corrected rule", document_type="errata"),
        ]
        ctx = _make_ctx(retrieval_results=results)
        messages = build_answer_synthesis_messages(ctx)
        user_msg = messages[1]["content"]

        assert "core_rules" in user_msg
        assert "errata" in user_msg


# ---------------------------------------------------------------------------
# Tests: Insufficient Knowledge Acknowledgement (AC-604)
# ---------------------------------------------------------------------------


class TestInsufficientKnowledge:
    """Tests for insufficient knowledge handling."""

    def test_insufficient_knowledge_prompt_instruction(self) -> None:
        """When knowledge is insufficient, prompt instructs acknowledgement."""
        ctx = _make_ctx(
            retrieval_results=[],
            knowledge_insufficient=True,
        )
        messages = build_answer_synthesis_messages(ctx)
        user_msg = messages[1]["content"]

        assert "No relevant knowledge" in user_msg

    def test_sufficient_knowledge_includes_sources(self) -> None:
        """When knowledge is available, sources are included in the prompt."""
        results = [_make_retrieval_result()]
        ctx = _make_ctx(retrieval_results=results)
        messages = build_answer_synthesis_messages(ctx)
        user_msg = messages[1]["content"]

        assert "Core Rulebook" in user_msg
        assert "flanking" in user_msg.lower()


# ---------------------------------------------------------------------------
# Tests: Prompt Context Inclusion
# ---------------------------------------------------------------------------


class TestPromptContext:
    """Tests that the prompt includes all required context."""

    def test_prompt_includes_question(self) -> None:
        """The player's question appears in the user message."""
        ctx = _make_ctx(question="How does flanking work in 5e?")
        messages = build_answer_synthesis_messages(ctx)
        user_msg = messages[1]["content"]

        assert "How does flanking work in 5e?" in user_msg

    def test_prompt_includes_player_count(self) -> None:
        """Player count appears in the system prompt when set."""
        ctx = _make_ctx(player_count=6)
        messages = build_answer_synthesis_messages(ctx)
        system_msg = messages[0]["content"]

        assert "6 players" in system_msg

    def test_prompt_omits_player_count_when_none(self) -> None:
        """Player count line is omitted when player_count is None."""
        ctx = _make_ctx(player_count=None)
        messages = build_answer_synthesis_messages(ctx)
        system_msg = messages[0]["content"]

        assert "players in this session" not in system_msg

    def test_prompt_includes_conversation_history(self) -> None:
        """Conversation history messages appear in the user prompt."""
        msg1 = MagicMock()
        msg1.type = "user_question"
        msg1.content = "What is AC?"
        msg2 = MagicMock()
        msg2.type = "ai_answer"
        msg2.content = "AC stands for Armor Class."

        ctx = _make_ctx(conversation_history=[msg1, msg2])
        messages = build_answer_synthesis_messages(ctx)
        user_msg = messages[1]["content"]

        assert "What is AC?" in user_msg
        assert "AC stands for Armor Class." in user_msg

    def test_prompt_includes_ad_hoc_text_context(self) -> None:
        """Ad-hoc text context appears in the user prompt."""
        ctx = _make_ctx(ad_hoc_text_context="The rogue is hidden behind a pillar")
        messages = build_answer_synthesis_messages(ctx)
        user_msg = messages[1]["content"]

        assert "The rogue is hidden behind a pillar" in user_msg

    def test_prompt_includes_image_descriptions(self) -> None:
        """Image descriptions appear in the user prompt."""
        ctx = _make_ctx(ad_hoc_image_descriptions=["A battle map with tokens"])
        messages = build_answer_synthesis_messages(ctx)
        user_msg = messages[1]["content"]

        assert "A battle map with tokens" in user_msg

    def test_prompt_includes_traversal_results(self) -> None:
        """Traversal results appear in the knowledge section."""
        traversal = [
            _make_traversal_result(content="Opportunity attacks", relationship="related_to"),
        ]
        ctx = _make_ctx(traversal_results=traversal)
        messages = build_answer_synthesis_messages(ctx)
        user_msg = messages[1]["content"]

        assert "Opportunity attacks" in user_msg
        assert "Related concepts" in user_msg

    def test_prompt_includes_source_metadata(self) -> None:
        """Source metadata (document name, section, page) in knowledge section."""
        results = [
            _make_retrieval_result(
                document_name="Core Rulebook",
                section_path="Chapter 9: Combat",
                page_number=42,
            ),
        ]
        ctx = _make_ctx(retrieval_results=results)
        messages = build_answer_synthesis_messages(ctx)
        user_msg = messages[1]["content"]

        assert "Core Rulebook" in user_msg
        assert "Chapter 9: Combat" in user_msg
        assert "p.42" in user_msg

    def test_prompt_has_system_and_user_messages(self) -> None:
        """Prompt has exactly two messages: system and user."""
        ctx = _make_ctx()
        messages = build_answer_synthesis_messages(ctx)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


# ---------------------------------------------------------------------------
# Tests: Error Propagation
# ---------------------------------------------------------------------------


class TestErrorPropagation:
    """Tests that model client errors propagate correctly."""

    async def test_model_client_error_propagates(self) -> None:
        """Exceptions from ModelClient.stream propagate unchanged."""
        ctx = _make_ctx()

        async def failing_stream(
            capability: Any,
            messages: Any,
            attribution: Any,
            **kwargs: Any,
        ) -> Any:
            raise RuntimeError("model exploded")
            yield  # pragma: no cover

        client = AsyncMock()
        client.stream = failing_stream  # type: ignore[assignment]
        stage = AnswerSynthesisStage()

        with pytest.raises(RuntimeError, match="model exploded"):
            async for _ in stage.execute(ctx, client):
                pass

    async def test_partial_stream_error_preserves_partial_text(self) -> None:
        """If error occurs mid-stream, accumulated text is NOT written.

        The stage accumulates tokens in a local list and only writes to
        ctx.answer_text after the stream completes. If the stream errors
        mid-way, ctx.answer_text remains empty.
        """
        ctx = _make_ctx()

        async def partial_then_error(
            capability: Any,
            messages: Any,
            attribution: Any,
            **kwargs: Any,
        ) -> Any:
            yield "partial "
            yield "answer "
            raise RuntimeError("mid-stream failure")

        client = AsyncMock()
        client.stream = partial_then_error  # type: ignore[assignment]
        stage = AnswerSynthesisStage()

        with pytest.raises(RuntimeError, match="mid-stream failure"):
            async for _ in stage.execute(ctx, client):
                pass

        # answer_text should remain empty since stream didn't complete
        assert ctx.answer_text == ""
