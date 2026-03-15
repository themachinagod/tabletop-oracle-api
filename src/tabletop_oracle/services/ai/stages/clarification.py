"""Stage 3: Clarification for the query pipeline (conditional).

When intent analysis flags ambiguity, this stage generates a clarification
question, persists it as an ``ai_clarification`` message, and signals
the pipeline to terminate. The pipeline resumes when the player responds.

Clarification round counting is stateless: derived by counting
``ai_clarification`` messages in the conversation thread (linked via
``in_reply_to_id``). After 2 rounds, the stage forces best-effort
processing by clearing the ambiguity flag.

Design reference: EPIC-004 design.md, Stage 3: Clarification.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tabletop_oracle.models.enums import MessageType, ModelCapability
from tabletop_oracle.services.ai.prompts.clarification import build_clarification_messages

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.ext.asyncio import AsyncSession

    from tabletop_oracle.services.ai.context import PipelineContext
    from tabletop_oracle.services.model.client import ModelClient
    from tabletop_oracle.streaming.events import SseEvent

logger = logging.getLogger(__name__)

#: Maximum number of clarification rounds before best-effort answer.
MAX_CLARIFICATION_ROUNDS = 2


class ClarificationStage:
    """Conditional clarification stage for the query pipeline.

    If the question is not ambiguous, this stage is a no-op pass-through.
    If ambiguous, it counts prior clarification rounds from the conversation
    thread. After ``MAX_CLARIFICATION_ROUNDS``, it clears the ambiguity
    flag so the pipeline proceeds with a best-effort answer. Otherwise,
    it generates and persists a clarification question.

    The ``pipeline_should_stop`` attribute is set to ``True`` when a
    clarification message has been persisted and the pipeline should
    terminate (the player needs to respond before processing can continue).

    Args:
        db: Async database session for persisting clarification messages
            and querying thread history.
    """

    def __init__(self, db: AsyncSession) -> None:
        self._db = db
        self.pipeline_should_stop: bool = False

    async def execute(
        self,
        ctx: PipelineContext,
        model_client: ModelClient,
    ) -> AsyncGenerator[SseEvent, None]:
        """Execute the clarification stage.

        If not ambiguous, returns immediately (no-op). If ambiguous,
        counts clarification rounds. After max rounds, clears ambiguity
        for best-effort. Otherwise generates and persists a clarification
        message.

        Args:
            ctx: Pipeline context with intent analysis results.
            model_client: Provider-agnostic model client for LLM calls.

        Yields:
            No SSE events. Pipeline orchestrator handles stream.step.
        """
        if not ctx.is_ambiguous:
            return
            yield  # pragma: no cover — makes this an async generator

        # Count clarification rounds from conversation thread
        round_count = _count_clarification_rounds(ctx)
        ctx.clarification_round = round_count

        if round_count >= MAX_CLARIFICATION_ROUNDS:
            logger.info(
                "Max clarification rounds reached, proceeding with best-effort",
                extra={
                    "message_id": str(ctx.user_message.id),
                    "clarification_round": round_count,
                },
            )
            ctx.is_ambiguous = False
            return
            yield  # pragma: no cover

        # Generate clarification question
        clarification_text = await _generate_clarification(ctx, model_client)

        # Persist the clarification message
        await _persist_clarification_message(
            db=self._db,
            ctx=ctx,
            content=clarification_text,
        )

        self.pipeline_should_stop = True

        logger.info(
            "Clarification message persisted",
            extra={
                "message_id": str(ctx.user_message.id),
                "clarification_round": round_count + 1,
            },
        )

        return
        yield  # pragma: no cover

    def reset(self) -> None:
        """Reset the stop signal for reuse across pipeline invocations."""
        self.pipeline_should_stop = False


def _count_clarification_rounds(ctx: PipelineContext) -> int:
    """Count clarification rounds from the conversation thread.

    Counts messages of type ``ai_clarification`` in the conversation
    history. This is the stateless, idempotent approach specified by
    the design — round count is derived from the thread, not stored
    as separate state.

    Args:
        ctx: Pipeline context with conversation_history populated.

    Returns:
        Number of prior clarification rounds.
    """
    return sum(
        1
        for msg in ctx.conversation_history
        if getattr(msg, "type", None) == MessageType.AI_CLARIFICATION
    )


async def _generate_clarification(
    ctx: PipelineContext,
    model_client: ModelClient,
) -> str:
    """Generate a clarification question using the clarification_generation model slot.

    Falls back to the suggested_clarification from intent analysis if
    the model call fails, and to a generic fallback if neither is available.

    Args:
        ctx: Pipeline context with intent analysis results.
        model_client: Provider-agnostic model client.

    Returns:
        The clarification question text.
    """
    from tabletop_oracle.services.model.token_usage import TokenAttribution

    messages = build_clarification_messages(ctx)
    attribution = TokenAttribution(
        user_id=ctx.session.user_id,
        session_id=ctx.session.id,
        message_id=ctx.user_message.id,
    )

    try:
        result = await model_client.complete(
            capability=ModelCapability.CLARIFICATION_GENERATION,
            messages=messages,
            attribution=attribution,
        )
        ctx.token_usage += result.input_tokens + result.output_tokens
        ctx.model_call_count += 1
        return result.content.strip()
    except Exception:
        logger.warning(
            "Clarification generation failed, using fallback",
            extra={"message_id": str(ctx.user_message.id)},
            exc_info=True,
        )
        if ctx.suggested_clarification:
            return ctx.suggested_clarification
        return (
            "Could you please clarify your question? "
            "I want to make sure I give you the right answer."
        )


async def _persist_clarification_message(
    db: AsyncSession,
    ctx: PipelineContext,
    content: str,
) -> None:
    """Persist an ai_clarification message to the database.

    Creates a new Message with type ``ai_clarification``, content set
    to the clarification question, and ``in_reply_to_id`` pointing to
    the user's original question.

    The sequence number is derived from the conversation history length
    plus one (the user's message that triggered clarification).

    Args:
        db: Async database session.
        ctx: Pipeline context with session and message data.
        content: The clarification question text.
    """
    from tabletop_oracle.models.message import Message

    # Sequence is one past the current highest in context
    next_sequence = ctx.user_message.sequence + 1

    clarification_msg = Message(
        session_id=ctx.session.id,
        type=MessageType.AI_CLARIFICATION,
        content=content,
        sequence=next_sequence,
        in_reply_to_id=ctx.user_message.id,
    )

    db.add(clarification_msg)
    await db.flush()

    logger.debug(
        "Clarification message created",
        extra={
            "clarification_message_id": str(clarification_msg.id),
            "in_reply_to_id": str(ctx.user_message.id),
            "session_id": str(ctx.session.id),
        },
    )
