"""Query pipeline orchestrator.

Executes stages sequentially, yielding SSE events for streaming. Checks
cancellation between stages and enforces pre-query guardrails before
pipeline execution begins. Handles errors gracefully, including partial
answer assembly when retrieval succeeds but synthesis fails.

Design reference: EPIC-004 design.md, QueryPipeline section.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from tabletop_oracle.streaming.payloads import (
    StreamCancelledPayload,
    StreamCompletedPayload,
    StreamErrorPayload,
    StreamStartedPayload,
    StreamStepPayload,
    payload_to_sse_event,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from tabletop_oracle.services.ai.context import PipelineContext
    from tabletop_oracle.services.model.client import ModelClient
    from tabletop_oracle.services.model.guardrail import GuardrailServiceProtocol
    from tabletop_oracle.streaming.events import SseEvent

logger = logging.getLogger(__name__)

# Stage labels shown to the user as stream.step events
_STAGE_LABELS: list[str] = [
    "Preparing context",
    "Analysing question",
    "Generating clarification",
    "Retrieving knowledge",
    "Synthesising answer",
    "Assembling citations",
    "Scoring confidence",
]


@runtime_checkable
class PipelineStage(Protocol):
    """Protocol for pipeline stages.

    Each stage receives the pipeline context, performs its work,
    mutates the context with its outputs, and yields SSE events.
    """

    def execute(
        self,
        ctx: PipelineContext,
        model_client: ModelClient,
    ) -> AsyncGenerator[SseEvent, None]:
        """Execute this pipeline stage.

        Implementations use ``async def`` with ``yield`` to produce an
        async generator. The protocol signature uses ``def`` (not
        ``async def``) because ``async def`` without ``yield`` is a
        coroutine wrapping an async generator, which breaks type
        compatibility.

        Args:
            ctx: Pipeline context with accumulated state.
            model_client: Provider-agnostic model client for LLM calls.

        Yields:
            SSE events to stream to the client.
        """
        ...


class QueryPipeline:
    """Orchestrates the query processing pipeline.

    Executes stages sequentially, yielding SSE events for streaming.
    Checks cancellation flag between stages. Each stage receives and
    mutates a PipelineContext that accumulates state across stages.

    Args:
        stages: Ordered list of pipeline stages to execute.
        model_client: Provider-agnostic LLM client.
        guardrail_service: Enforces token and query guardrails.
    """

    def __init__(
        self,
        stages: list[PipelineStage],
        model_client: ModelClient,
        guardrail_service: GuardrailServiceProtocol,
    ) -> None:
        self._stages = stages
        self._model_client = model_client
        self._guardrail_service = guardrail_service

    async def execute(
        self,
        ctx: PipelineContext,
        cancel_check: Callable[[], Awaitable[bool]],
    ) -> AsyncGenerator[SseEvent, None]:
        """Execute the query pipeline and yield SSE events.

        Flow:
        1. Yield ``stream.started``.
        2. Check pre-query guardrails. If exceeded, yield ``stream.error``.
        3. For each stage: yield ``stream.step``, execute, check cancellation.
        4. On completion, yield ``stream.completed``.
        5. On error, yield ``stream.error``. If retrieval succeeded but
           synthesis failed, yield a partial answer.

        Args:
            ctx: Pre-populated pipeline context with session, message, and
                 game metadata.
            cancel_check: Async callable that returns True if the query
                          has been cancelled by the player.

        Yields:
            SseEvent instances for the T002 streaming infrastructure.
        """
        message_id = ctx.user_message.id

        # 1. Stream started
        yield payload_to_sse_event(StreamStartedPayload(message_id=message_id))

        # 2. Pre-query guardrail check
        guardrail_result = await self._guardrail_service.check_pre_query(
            session_id=ctx.session.id,
        )
        if not guardrail_result.allowed:
            logger.warning(
                "Pre-query guardrail denied",
                extra={
                    "guardrail_type": guardrail_result.guardrail_type,
                    "session_id": str(ctx.session.id),
                    "message_id": str(message_id),
                },
            )
            yield payload_to_sse_event(
                StreamErrorPayload(
                    code="GUARDRAIL_EXCEEDED",
                    message=guardrail_result.message or "Usage limit exceeded",
                ),
            )
            return

        # 3. Execute stages
        synthesis_failed = False
        synthesis_stage_index = self._find_stage_index("Synthesising answer")

        for i, stage in enumerate(self._stages):
            # Check cancellation between stages
            if await cancel_check():
                logger.info(
                    "Query cancelled between stages",
                    extra={
                        "stage_index": i,
                        "message_id": str(message_id),
                    },
                )
                yield payload_to_sse_event(
                    StreamCancelledPayload(message_id=message_id),
                )
                return

            # Emit step indicator
            stage_label = _STAGE_LABELS[i] if i < len(_STAGE_LABELS) else f"Stage {i + 1}"
            yield payload_to_sse_event(
                StreamStepPayload(step=stage_label),
            )

            # Execute stage
            try:
                async for event in stage.execute(ctx, self._model_client):
                    yield event
            except Exception as exc:
                logger.exception(
                    "Pipeline stage failed",
                    extra={
                        "stage_index": i,
                        "stage_label": stage_label,
                        "message_id": str(message_id),
                    },
                )

                # Partial answer: if retrieval succeeded but synthesis failed
                if (
                    synthesis_stage_index is not None
                    and i >= synthesis_stage_index
                    and ctx.retrieval_results
                ):
                    synthesis_failed = True
                    async for event in self._emit_partial_answer(ctx):
                        yield event
                    return

                yield payload_to_sse_event(
                    StreamErrorPayload(
                        code="SERVICE_UNAVAILABLE",
                        message=_user_facing_error(exc),
                    ),
                )
                return

        # 4. Stream completed (only if synthesis didn't fail with partial)
        if not synthesis_failed:
            yield payload_to_sse_event(
                StreamCompletedPayload(message_id=message_id),
            )

    def _find_stage_index(self, label: str) -> int | None:
        """Find the index of a stage by its label.

        Args:
            label: The stage label to search for.

        Returns:
            The stage index, or None if the label is not in the stage list.
        """
        for i, stage_label in enumerate(_STAGE_LABELS):
            if stage_label == label:
                return i
        return None

    async def _emit_partial_answer(
        self,
        ctx: PipelineContext,
    ) -> AsyncGenerator[SseEvent, None]:
        """Emit a partial answer from retrieval results when synthesis fails.

        Assembles the top retrieval results as a simplified answer with
        source excerpts. Sets confidence to 0.0.

        Args:
            ctx: Pipeline context with populated retrieval_results.

        Yields:
            SSE events for the partial answer.
        """
        from tabletop_oracle.streaming.payloads import (
            StreamCitationPayload,
            StreamConfidencePayload,
            StreamTokenPayload,
        )

        logger.warning(
            "Emitting partial answer due to synthesis failure",
            extra={"message_id": str(ctx.user_message.id)},
        )

        # Build partial answer text from retrieval results
        partial_text = (
            "I encountered an issue generating a complete answer, "
            "but here are the relevant sources I found:\n\n"
        )
        for i, result in enumerate(ctx.retrieval_results[:5], start=1):
            source_name = result.source.get("document_name", "Unknown source")
            partial_text += f"[{i}] {source_name}: {result.content[:200]}\n\n"

        ctx.answer_text = partial_text
        ctx.confidence_score = 0.0

        # Emit tokens
        yield payload_to_sse_event(
            StreamStepPayload(step="Partial answer (synthesis failed)"),
        )
        yield payload_to_sse_event(StreamTokenPayload(token=partial_text))

        # Emit citations from retrieval results
        for i, result in enumerate(ctx.retrieval_results[:5], start=1):
            yield payload_to_sse_event(
                StreamCitationPayload(
                    index=i,
                    document_name=result.source.get("document_name", "Unknown"),
                    section=result.source.get("section", ""),
                    excerpt=result.content[:200],
                ),
            )

        # Emit zero confidence
        yield payload_to_sse_event(
            StreamConfidencePayload(score=0.0, label="Partial answer"),
        )

        # Complete the stream
        yield payload_to_sse_event(
            StreamCompletedPayload(message_id=ctx.user_message.id),
        )


def _user_facing_error(exc: Exception) -> str:
    """Build a human-readable error message from an exception.

    Avoids leaking internal details to the client while providing
    enough context for the player to understand what happened.

    Args:
        exc: The caught exception.

    Returns:
        A player-friendly error message.
    """
    from tabletop_oracle.errors.exceptions import GuardrailExceededError, ModelUnavailableError

    if isinstance(exc, GuardrailExceededError):
        return exc.message
    if isinstance(exc, ModelUnavailableError):
        return "The AI service is temporarily unavailable. Please try again in a moment."
    return "An unexpected error occurred while processing your question. Please try again."
