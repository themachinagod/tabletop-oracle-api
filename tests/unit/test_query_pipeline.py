"""Unit tests for QueryPipeline, PipelineContext, and PipelineStage.

Covers:
- Pipeline executes stages in order
- Cancellation detected between stages (AC-611)
- SSE events yielded correctly (stream.started, stream.step, stream.completed)
- Pre-query guardrails checked before execution
- Error handling wraps failures into stream.error
- Partial answer on synthesis failure
- PipelineContext dataclass construction and mutation
- PipelineStage protocol conformance
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tabletop_oracle.services.ai.context import (
    CitationData,
    PipelineContext,
    RetrievalResult,
)
from tabletop_oracle.services.ai.pipeline import (
    PipelineStage,
    QueryPipeline,
    _user_facing_error,
)
from tabletop_oracle.services.model.types import GuardrailCheckResult
from tabletop_oracle.streaming.events import SseEvent

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

_SESSION_ID = uuid.uuid4()
_MESSAGE_ID = uuid.uuid4()
_GAME_ID = uuid.uuid4()


def _make_session(
    session_id: uuid.UUID = _SESSION_ID,
    game_id: uuid.UUID = _GAME_ID,
) -> MagicMock:
    """Build a mock Session object."""
    session = MagicMock()
    session.id = session_id
    session.game_id = game_id
    session.player_count = 4
    return session


def _make_message(
    message_id: uuid.UUID = _MESSAGE_ID,
    content: str = "How does ranged combat work?",
) -> MagicMock:
    """Build a mock Message object."""
    msg = MagicMock()
    msg.id = message_id
    msg.content = content
    return msg


def _make_context(
    session: MagicMock | None = None,
    message: MagicMock | None = None,
) -> PipelineContext:
    """Build a PipelineContext with sensible defaults."""
    return PipelineContext(
        session=session or _make_session(),
        user_message=message or _make_message(),
        game_id=_GAME_ID,
        active_expansion_ids=[],
        player_count=4,
    )


class MockStage:
    """A mock pipeline stage that records calls and yields configurable events.

    Args:
        name: Identifier for the stage (for assertion clarity).
        events: SSE events to yield when executed.
        side_effect: Exception to raise during execution (simulates failure).
        mutate_ctx: Callable to mutate the context during execution.
    """

    def __init__(
        self,
        name: str = "mock_stage",
        events: list[SseEvent] | None = None,
        side_effect: Exception | None = None,
        mutate_ctx: Any | None = None,
    ) -> None:
        self.name = name
        self.events = events or []
        self.side_effect = side_effect
        self.mutate_ctx = mutate_ctx
        self.executed = False
        self.call_count = 0

    async def execute(
        self,
        ctx: PipelineContext,
        model_client: Any,
    ) -> AsyncGenerator[SseEvent, None]:
        """Execute the mock stage."""
        self.executed = True
        self.call_count += 1

        if self.mutate_ctx is not None:
            self.mutate_ctx(ctx)

        if self.side_effect is not None:
            raise self.side_effect

        for event in self.events:
            yield event


def _make_guardrail_service(allowed: bool = True, message: str | None = None) -> MagicMock:
    """Build a mock GuardrailServiceProtocol."""
    service = MagicMock()
    result = GuardrailCheckResult(
        allowed=allowed,
        message=message,
        guardrail_type="per_session_query" if not allowed else None,
    )
    service.check_pre_query = AsyncMock(return_value=result)
    return service


def _make_model_client() -> MagicMock:
    """Build a mock ModelClient."""
    return MagicMock()


def _never_cancelled() -> AsyncMock:
    """Cancel check that always returns False."""
    return AsyncMock(return_value=False)


def _cancel_after(n: int) -> AsyncMock:
    """Cancel check that returns True after n calls."""
    counter = {"count": 0}

    async def _check() -> bool:
        counter["count"] += 1
        return counter["count"] > n

    return AsyncMock(side_effect=_check)


async def _collect_events(
    gen: AsyncGenerator[SseEvent, None],
) -> list[SseEvent]:
    """Drain an async generator into a list."""
    events: list[SseEvent] = []
    async for event in gen:
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# PipelineContext tests
# ---------------------------------------------------------------------------


class TestPipelineContext:
    """Tests for PipelineContext dataclass."""

    def test_construction_sets_defaults_for_optional_fields(self) -> None:
        """Verify defaults are set for optional fields."""
        ctx = _make_context()

        assert ctx.game_id == _GAME_ID
        assert ctx.player_count == 4
        assert ctx.conversation_history == []
        assert ctx.ad_hoc_text_context is None
        assert ctx.ad_hoc_image_descriptions == []
        assert ctx.intent is None
        assert ctx.is_ambiguous is False
        assert ctx.clarification_round == 0
        assert ctx.retrieval_results == []
        assert ctx.traversal_results == []
        assert ctx.answer_text == ""
        assert ctx.citations == []
        assert ctx.confidence_score is None
        assert ctx.token_usage == 0
        assert ctx.model_call_count == 0

    def test_mutation_updates_fields(self) -> None:
        """Verify PipelineContext is mutable (stages write to it)."""
        ctx = _make_context()

        ctx.intent = "ranged combat rules"
        ctx.is_ambiguous = True
        ctx.clarification_round = 1
        ctx.answer_text = "Here is the answer."
        ctx.confidence_score = 0.85
        ctx.token_usage = 1500
        ctx.model_call_count = 3

        assert ctx.intent == "ranged combat rules"
        assert ctx.is_ambiguous is True
        assert ctx.clarification_round == 1
        assert ctx.answer_text == "Here is the answer."
        assert ctx.confidence_score == 0.85
        assert ctx.token_usage == 1500
        assert ctx.model_call_count == 3

    def test_retrieval_results_default_factory_independent_lists(self) -> None:
        """Verify each PipelineContext has independent list instances."""
        ctx1 = _make_context()
        ctx2 = _make_context()

        concept_id = uuid.uuid4()
        ctx1.retrieval_results.append(
            RetrievalResult(concept_id=concept_id, content="test", score=0.9),
        )
        assert len(ctx2.retrieval_results) == 0


class TestCitationData:
    """Tests for CitationData dataclass."""

    def test_construction_with_all_fields(self) -> None:
        """Verify all fields are set correctly."""
        citation = CitationData(
            index=1,
            document_name="Core Rules v2",
            document_type="core_rules",
            section="Chapter 5.3",
            page_number=42,
            excerpt="Ranged attacks use...",
        )
        assert citation.index == 1
        assert citation.document_name == "Core Rules v2"
        assert citation.page_number == 42

    def test_optional_fields_default_to_empty(self) -> None:
        """Verify optional fields default correctly."""
        citation = CitationData(index=1, document_name="Rules", document_type="core_rules")
        assert citation.section == ""
        assert citation.page_number is None
        assert citation.excerpt == ""


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_construction_with_source_dict(self) -> None:
        """Verify source dict can be populated."""
        result = RetrievalResult(
            concept_id=uuid.uuid4(),
            content="ranged combat rules",
            score=0.95,
            source={"document_name": "Core Rules", "section": "5.3"},
        )
        assert result.source["document_name"] == "Core Rules"

    def test_source_defaults_to_empty_dict(self) -> None:
        """Verify source defaults to empty dict."""
        result = RetrievalResult(concept_id=uuid.uuid4(), content="test", score=0.5)
        assert result.source == {}


# ---------------------------------------------------------------------------
# PipelineStage protocol tests
# ---------------------------------------------------------------------------


class TestPipelineStageProtocol:
    """Tests for PipelineStage protocol conformance."""

    def test_mock_stage_satisfies_protocol(self) -> None:
        """Verify MockStage satisfies the PipelineStage protocol."""
        stage = MockStage()
        assert isinstance(stage, PipelineStage)

    def test_object_without_execute_does_not_match(self) -> None:
        """Verify objects without execute() do not match the protocol."""

        class NotAStage:
            pass

        assert not isinstance(NotAStage(), PipelineStage)


# ---------------------------------------------------------------------------
# QueryPipeline tests
# ---------------------------------------------------------------------------


class TestQueryPipelineExecution:
    """Tests for QueryPipeline.execute() happy path and stage ordering."""

    @pytest.fixture()
    def pipeline_components(self) -> dict[str, Any]:
        """Create reusable pipeline components."""
        stages = [MockStage(f"stage_{i}") for i in range(7)]
        return {
            "stages": stages,
            "model_client": _make_model_client(),
            "guardrail_service": _make_guardrail_service(),
            "ctx": _make_context(),
            "cancel_check": _never_cancelled(),
        }

    async def test_seven_stages_all_executed_in_order(
        self,
        pipeline_components: dict[str, Any],
    ) -> None:
        """Verify all 7 stages execute in order."""
        pipeline = QueryPipeline(
            stages=pipeline_components["stages"],
            model_client=pipeline_components["model_client"],
            guardrail_service=pipeline_components["guardrail_service"],
        )

        await _collect_events(
            pipeline.execute(
                pipeline_components["ctx"],
                pipeline_components["cancel_check"],
            ),
        )

        for stage in pipeline_components["stages"]:
            assert stage.executed, f"{stage.name} was not executed"
            assert stage.call_count == 1

    async def test_seven_stages_yields_started_and_completed(
        self,
        pipeline_components: dict[str, Any],
    ) -> None:
        """Verify stream.started and stream.completed are yielded."""
        pipeline = QueryPipeline(
            stages=pipeline_components["stages"],
            model_client=pipeline_components["model_client"],
            guardrail_service=pipeline_components["guardrail_service"],
        )

        events = await _collect_events(
            pipeline.execute(
                pipeline_components["ctx"],
                pipeline_components["cancel_check"],
            ),
        )

        assert events[0].type == "stream.started"
        assert events[-1].type == "stream.completed"

    async def test_seven_stages_yields_step_per_stage(
        self,
        pipeline_components: dict[str, Any],
    ) -> None:
        """Verify a stream.step event is emitted before each stage."""
        pipeline = QueryPipeline(
            stages=pipeline_components["stages"],
            model_client=pipeline_components["model_client"],
            guardrail_service=pipeline_components["guardrail_service"],
        )

        events = await _collect_events(
            pipeline.execute(
                pipeline_components["ctx"],
                pipeline_components["cancel_check"],
            ),
        )

        step_events = [e for e in events if e.type == "stream.step"]
        assert len(step_events) == 7

    async def test_stage_events_passed_through(
        self,
        pipeline_components: dict[str, Any],
    ) -> None:
        """Verify events from stages are yielded through the pipeline."""
        token_event = SseEvent(type="stream.token", payload={"token": "hello"})
        pipeline_components["stages"][4] = MockStage("synthesis", events=[token_event])

        pipeline = QueryPipeline(
            stages=pipeline_components["stages"],
            model_client=pipeline_components["model_client"],
            guardrail_service=pipeline_components["guardrail_service"],
        )

        events = await _collect_events(
            pipeline.execute(
                pipeline_components["ctx"],
                pipeline_components["cancel_check"],
            ),
        )

        assert token_event in events


class TestQueryPipelineCancellation:
    """Tests for cancellation detection between stages (AC-611)."""

    async def test_cancel_before_first_stage_yields_cancelled(self) -> None:
        """Verify cancellation before any stage yields stream.cancelled."""
        cancel = AsyncMock(return_value=True)
        stages = [MockStage("stage_0")]

        pipeline = QueryPipeline(
            stages=stages,
            model_client=_make_model_client(),
            guardrail_service=_make_guardrail_service(),
        )

        events = await _collect_events(
            pipeline.execute(_make_context(), cancel),
        )

        assert events[0].type == "stream.started"
        assert events[1].type == "stream.cancelled"
        assert events[1].payload["message_id"] == str(_MESSAGE_ID)
        assert not stages[0].executed

    async def test_cancel_after_second_stage_only_first_two_executed(self) -> None:
        """Verify stages after cancellation are not executed."""
        cancel = _cancel_after(2)
        stages = [MockStage(f"stage_{i}") for i in range(5)]

        pipeline = QueryPipeline(
            stages=stages,
            model_client=_make_model_client(),
            guardrail_service=_make_guardrail_service(),
        )

        events = await _collect_events(
            pipeline.execute(_make_context(), cancel),
        )

        assert stages[0].executed
        assert stages[1].executed
        assert not stages[2].executed
        cancelled_events = [e for e in events if e.type == "stream.cancelled"]
        assert len(cancelled_events) == 1


class TestQueryPipelineGuardrails:
    """Tests for pre-query guardrail checking."""

    async def test_guardrail_denied_yields_error_and_stops(self) -> None:
        """Verify denied guardrail yields stream.error and stops pipeline."""
        guardrail = _make_guardrail_service(
            allowed=False,
            message="This session has reached its question limit.",
        )
        stages = [MockStage("stage_0")]

        pipeline = QueryPipeline(
            stages=stages,
            model_client=_make_model_client(),
            guardrail_service=guardrail,
        )

        events = await _collect_events(
            pipeline.execute(_make_context(), _never_cancelled()),
        )

        assert events[0].type == "stream.started"
        assert events[1].type == "stream.error"
        assert events[1].payload["code"] == "GUARDRAIL_EXCEEDED"
        assert "question limit" in events[1].payload["message"]
        assert not stages[0].executed

    async def test_guardrail_allowed_continues_pipeline(self) -> None:
        """Verify pipeline proceeds when guardrails pass."""
        guardrail = _make_guardrail_service(allowed=True)
        stages = [MockStage("stage_0")]

        pipeline = QueryPipeline(
            stages=stages,
            model_client=_make_model_client(),
            guardrail_service=guardrail,
        )

        await _collect_events(
            pipeline.execute(_make_context(), _never_cancelled()),
        )

        assert stages[0].executed
        guardrail.check_pre_query.assert_called_once()


class TestQueryPipelineErrorHandling:
    """Tests for error handling and stream.error emission."""

    async def test_stage_failure_early_yields_error_event(self) -> None:
        """Verify stage failure emits stream.error with safe message."""
        failing_stage = MockStage("intent", side_effect=RuntimeError("LLM timeout"))
        stages = [MockStage("context_prep"), failing_stage, MockStage("retrieval")]

        pipeline = QueryPipeline(
            stages=stages,
            model_client=_make_model_client(),
            guardrail_service=_make_guardrail_service(),
        )

        events = await _collect_events(
            pipeline.execute(_make_context(), _never_cancelled()),
        )

        error_events = [e for e in events if e.type == "stream.error"]
        assert len(error_events) == 1
        assert error_events[0].payload["code"] == "SERVICE_UNAVAILABLE"
        # Verify later stages did not execute
        assert not stages[2].executed

    async def test_stage_failure_no_completed_event(self) -> None:
        """Verify stream.completed is NOT emitted on failure."""
        stages = [MockStage("failing", side_effect=ValueError("bad"))]

        pipeline = QueryPipeline(
            stages=stages,
            model_client=_make_model_client(),
            guardrail_service=_make_guardrail_service(),
        )

        events = await _collect_events(
            pipeline.execute(_make_context(), _never_cancelled()),
        )

        completed_events = [e for e in events if e.type == "stream.completed"]
        assert len(completed_events) == 0


class TestQueryPipelinePartialAnswer:
    """Tests for partial answer when retrieval succeeds but synthesis fails."""

    async def test_synthesis_fails_with_retrieval_emits_partial(self) -> None:
        """Verify partial answer is emitted when synthesis fails after retrieval."""
        concept_id = uuid.uuid4()

        def populate_retrieval(ctx: PipelineContext) -> None:
            ctx.retrieval_results = [
                RetrievalResult(
                    concept_id=concept_id,
                    content="Ranged attacks use line of sight.",
                    score=0.95,
                    source={"document_name": "Core Rules", "section": "5.3"},
                ),
            ]

        # Stages: context_prep, intent, clarification, retrieval (populates),
        #         synthesis (fails), citation, confidence
        stages = [
            MockStage("context_prep"),
            MockStage("intent"),
            MockStage("clarification"),
            MockStage("retrieval", mutate_ctx=populate_retrieval),
            MockStage("synthesis", side_effect=RuntimeError("Model failed")),
            MockStage("citation"),
            MockStage("confidence"),
        ]

        pipeline = QueryPipeline(
            stages=stages,
            model_client=_make_model_client(),
            guardrail_service=_make_guardrail_service(),
        )

        events = await _collect_events(
            pipeline.execute(_make_context(), _never_cancelled()),
        )

        # Should have partial answer events instead of error
        event_types = [e.type for e in events]
        assert "stream.token" in event_types
        assert "stream.citation" in event_types
        assert "stream.confidence" in event_types
        assert "stream.completed" in event_types

        # Confidence should be 0.0
        conf_event = next(e for e in events if e.type == "stream.confidence")
        assert conf_event.payload["score"] == 0.0

    async def test_synthesis_fails_no_retrieval_emits_error(self) -> None:
        """Verify error (not partial) when synthesis fails without retrieval."""
        stages = [
            MockStage("context_prep"),
            MockStage("intent"),
            MockStage("clarification"),
            MockStage("retrieval"),  # No retrieval results populated
            MockStage("synthesis", side_effect=RuntimeError("Model failed")),
            MockStage("citation"),
            MockStage("confidence"),
        ]

        pipeline = QueryPipeline(
            stages=stages,
            model_client=_make_model_client(),
            guardrail_service=_make_guardrail_service(),
        )

        events = await _collect_events(
            pipeline.execute(_make_context(), _never_cancelled()),
        )

        error_events = [e for e in events if e.type == "stream.error"]
        assert len(error_events) == 1

    async def test_early_stage_failure_no_partial_answer(self) -> None:
        """Verify stages before synthesis don't trigger partial answer."""
        stages = [
            MockStage("context_prep", side_effect=RuntimeError("DB error")),
            MockStage("intent"),
        ]

        pipeline = QueryPipeline(
            stages=stages,
            model_client=_make_model_client(),
            guardrail_service=_make_guardrail_service(),
        )

        events = await _collect_events(
            pipeline.execute(_make_context(), _never_cancelled()),
        )

        error_events = [e for e in events if e.type == "stream.error"]
        assert len(error_events) == 1
        token_events = [e for e in events if e.type == "stream.token"]
        assert len(token_events) == 0


# ---------------------------------------------------------------------------
# Error message helper tests
# ---------------------------------------------------------------------------


class TestUserFacingError:
    """Tests for the _user_facing_error helper."""

    def test_guardrail_exceeded_returns_original_message(self) -> None:
        """Verify GuardrailExceededError returns its own message."""
        from tabletop_oracle.errors.exceptions import GuardrailExceededError

        exc = GuardrailExceededError("Session limit reached", guardrail_type="per_session_query")
        assert _user_facing_error(exc) == "Session limit reached"

    def test_model_unavailable_returns_safe_message(self) -> None:
        """Verify ModelUnavailableError returns safe message."""
        from tabletop_oracle.errors.exceptions import ModelUnavailableError

        exc = ModelUnavailableError("gpt-4 timed out", capability="answer_synthesis")
        result = _user_facing_error(exc)
        assert "temporarily unavailable" in result

    def test_generic_exception_returns_safe_message(self) -> None:
        """Verify generic exceptions don't leak details."""
        exc = RuntimeError("SQL injection attempt detected")
        result = _user_facing_error(exc)
        assert "SQL" not in result
        assert "unexpected error" in result


# ---------------------------------------------------------------------------
# Integration-style test: full pipeline flow
# ---------------------------------------------------------------------------


class TestQueryPipelineFullFlow:
    """Integration test for the full 7-stage pipeline flow."""

    async def test_full_pipeline_seven_stages_complete_event_sequence(self) -> None:
        """Verify complete event sequence for a successful 7-stage run."""
        token_event = SseEvent(type="stream.token", payload={"token": "Answer text"})
        citation_event = SseEvent(
            type="stream.citation",
            payload={"index": 1, "document_name": "Rules", "section": "", "excerpt": ""},
        )
        confidence_event = SseEvent(
            type="stream.confidence",
            payload={"score": 0.85, "label": "High"},
        )

        stages = [
            MockStage("context_prep"),
            MockStage("intent"),
            MockStage("clarification"),
            MockStage("retrieval"),
            MockStage("synthesis", events=[token_event]),
            MockStage("citation", events=[citation_event]),
            MockStage("confidence", events=[confidence_event]),
        ]

        pipeline = QueryPipeline(
            stages=stages,
            model_client=_make_model_client(),
            guardrail_service=_make_guardrail_service(),
        )

        events = await _collect_events(
            pipeline.execute(_make_context(), _never_cancelled()),
        )

        event_types = [e.type for e in events]

        assert event_types[0] == "stream.started"
        assert event_types[-1] == "stream.completed"
        assert event_types.count("stream.step") == 7
        assert "stream.token" in event_types
        assert "stream.citation" in event_types
        assert "stream.confidence" in event_types

        # Total: 1 started + 7 steps + 3 stage events + 1 completed = 12
        assert len(events) == 12

    async def test_empty_pipeline_yields_started_and_completed(self) -> None:
        """Verify empty pipeline still yields started and completed."""
        pipeline = QueryPipeline(
            stages=[],
            model_client=_make_model_client(),
            guardrail_service=_make_guardrail_service(),
        )

        events = await _collect_events(
            pipeline.execute(_make_context(), _never_cancelled()),
        )

        assert len(events) == 2
        assert events[0].type == "stream.started"
        assert events[1].type == "stream.completed"

    async def test_message_id_correct_in_lifecycle_events(self) -> None:
        """Verify message_id is correct in started, completed, cancelled events."""
        msg_id = uuid.uuid4()
        ctx = _make_context(message=_make_message(message_id=msg_id))

        pipeline = QueryPipeline(
            stages=[MockStage("stage_0")],
            model_client=_make_model_client(),
            guardrail_service=_make_guardrail_service(),
        )

        events = await _collect_events(
            pipeline.execute(ctx, _never_cancelled()),
        )

        started = events[0]
        completed = events[-1]
        assert started.payload["message_id"] == str(msg_id)
        assert completed.payload["message_id"] == str(msg_id)
