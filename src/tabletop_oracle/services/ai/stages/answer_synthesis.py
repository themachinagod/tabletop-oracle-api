"""Stage 5: Answer synthesis for the query pipeline.

Generates a natural-language answer grounded in KG-retrieved knowledge
using the ``answer_synthesis`` model slot. Streams tokens via
ModelClient.stream() and yields ``stream.token`` SSE events. Embeds
numbered source markers ([1], [2]) for downstream citation mapping.

Design reference: EPIC-004 design.md, Stage 5: Answer Synthesis.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tabletop_oracle.models.enums import ModelCapability
from tabletop_oracle.services.ai.prompts.answer_synthesis import (
    build_answer_synthesis_messages,
)
from tabletop_oracle.streaming.payloads import StreamTokenPayload, payload_to_sse_event

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from tabletop_oracle.services.ai.context import PipelineContext
    from tabletop_oracle.services.model.client import ModelClient
    from tabletop_oracle.services.model.token_usage import TokenAttribution
    from tabletop_oracle.streaming.events import SseEvent

logger = logging.getLogger(__name__)


class AnswerSynthesisStage:
    """Generates a grounded answer from retrieved KG knowledge.

    Uses the ``answer_synthesis`` model slot (highest-quality model)
    via ``ModelClient.stream()``. Each token is yielded as a
    ``stream.token`` SSE event and accumulated in ``ctx.answer_text``.

    The prompt enforces:
    - Grounding in provided knowledge only, no fabrication (AC-601, AC-604).
    - Numbered source markers [1], [2] for citation mapping (AC-602).
    - Document type priority: Errata > FAQ > Core Rules > Other (AC-609).
    - Strategy content framed as guidance, not rules (AC-607).
    - Honest acknowledgement of insufficient knowledge.

    Populates on ``PipelineContext``:
    - ``answer_text``: Full accumulated answer text.
    - ``token_usage``: Incremented (best-effort from streaming).
    - ``model_call_count``: Incremented by 1.
    """

    async def execute(
        self,
        ctx: PipelineContext,
        model_client: ModelClient,
    ) -> AsyncGenerator[SseEvent, None]:
        """Generate the answer by streaming tokens from the LLM.

        Builds the synthesis prompt from retrieved knowledge and session
        context, streams tokens via the model client, yields each token
        as a ``stream.token`` SSE event, and accumulates the full answer
        in ``ctx.answer_text``.

        Args:
            ctx: Pipeline context with retrieval results and session data.
            model_client: Provider-agnostic model client for LLM calls.

        Yields:
            ``stream.token`` SSE events, one per token chunk.
        """
        messages = build_answer_synthesis_messages(ctx)
        attribution = _build_attribution(ctx)

        logger.info(
            "Starting answer synthesis",
            extra={
                "message_id": str(ctx.user_message.id),
                "retrieval_count": len(ctx.retrieval_results),
                "knowledge_insufficient": ctx.knowledge_insufficient,
            },
        )

        accumulated_tokens: list[str] = []

        async for token in model_client.stream(
            capability=ModelCapability.ANSWER_SYNTHESIS,
            messages=messages,
            attribution=attribution,
        ):
            accumulated_tokens.append(token)
            yield payload_to_sse_event(StreamTokenPayload(token=token))

        ctx.answer_text = "".join(accumulated_tokens)
        ctx.model_call_count += 1

        logger.info(
            "Answer synthesis complete",
            extra={
                "message_id": str(ctx.user_message.id),
                "answer_length": len(ctx.answer_text),
            },
        )


def _build_attribution(ctx: PipelineContext) -> TokenAttribution:
    """Build token attribution from the pipeline context.

    Args:
        ctx: Pipeline context with session and message identifiers.

    Returns:
        TokenAttribution for tracking token consumption.
    """
    from tabletop_oracle.services.model.token_usage import TokenAttribution

    return TokenAttribution(
        user_id=ctx.session.user_id,
        session_id=ctx.session.id,
        message_id=ctx.user_message.id,
    )
