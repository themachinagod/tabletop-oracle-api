"""Stage 2: Intent analysis for the query pipeline.

Analyses the player's question to determine intent and detect ambiguity
using the ``intent_analysis`` model slot. Parses a structured JSON
response and populates the pipeline context with intent, ambiguity
flag, and optional clarification suggestion.

Design reference: EPIC-004 design.md, Stage 2: Intent Analysis.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from tabletop_oracle.models.enums import ModelCapability
from tabletop_oracle.services.ai.prompts.intent import build_intent_messages

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from tabletop_oracle.services.ai.context import PipelineContext
    from tabletop_oracle.services.model.client import ModelClient
    from tabletop_oracle.services.model.token_usage import TokenAttribution
    from tabletop_oracle.streaming.events import SseEvent

logger = logging.getLogger(__name__)

# Expected keys in the LLM's JSON response.
_REQUIRED_KEYS = {"intent", "is_ambiguous"}


class IntentAnalysisStage:
    """Analyses the player's question to determine intent and detect ambiguity.

    Uses the ``intent_analysis`` model slot via ``ModelClient.complete()``.
    The prompt includes the user's question, conversation history, session
    configuration, and ad-hoc context. The LLM returns structured JSON
    with intent, ambiguity flag, reason, and suggested clarification.

    Populates on ``PipelineContext``:
    - ``intent``: Free-text summary of the player's intent.
    - ``is_ambiguous``: Whether clarification is needed.
    - ``token_usage``: Incremented by tokens consumed.
    - ``model_call_count``: Incremented by 1.
    """

    async def execute(
        self,
        ctx: PipelineContext,
        model_client: ModelClient,
    ) -> AsyncGenerator[SseEvent, None]:
        """Analyse the player's question for intent and ambiguity.

        Calls the LLM with the intent analysis prompt, parses the JSON
        response, and writes results to the pipeline context.

        Args:
            ctx: Pipeline context with session, message, and context data.
            model_client: Provider-agnostic model client for LLM calls.

        Yields:
            No SSE events. This stage is silent.

        Raises:
            ValueError: If the LLM response cannot be parsed as valid
                intent analysis JSON.
        """
        messages = build_intent_messages(ctx)
        attribution = _build_attribution(ctx)

        result = await model_client.complete(
            capability=ModelCapability.INTENT_ANALYSIS,
            messages=messages,
            attribution=attribution,
        )

        parsed = _parse_response(result.content)
        _apply_to_context(ctx, parsed, result)

        logger.info(
            "Intent analysis complete",
            extra={
                "message_id": str(ctx.user_message.id),
                "intent": parsed.get("intent", ""),
                "is_ambiguous": parsed.get("is_ambiguous", False),
            },
        )

        # Async generator must contain at least one yield (even if unreachable)
        # to satisfy the protocol's AsyncGenerator return type.
        return
        yield  # pragma: no cover — makes this function an async generator


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


def _parse_response(content: str) -> dict[str, Any]:
    """Parse the LLM response as intent analysis JSON.

    Handles responses that may be wrapped in markdown code fences.

    Args:
        content: Raw LLM response text.

    Returns:
        Parsed dict with intent analysis fields.

    Raises:
        ValueError: If the response is not valid JSON or is missing
            required keys.
    """
    cleaned = content.strip()

    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [
            line
            for i, line in enumerate(lines)
            if not (i == 0 or (i == len(lines) - 1 and line.strip() == "```"))
        ]
        cleaned = "\n".join(lines)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Intent analysis response is not valid JSON: {content[:200]}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"Intent analysis response is not a JSON object: {type(parsed).__name__}")

    missing = _REQUIRED_KEYS - set(parsed.keys())
    if missing:
        raise ValueError(f"Intent analysis response missing required keys: {missing}")

    return parsed


def _apply_to_context(
    ctx: PipelineContext,
    parsed: dict[str, Any],
    result: Any,
) -> None:
    """Apply parsed intent analysis results to the pipeline context.

    Args:
        ctx: Pipeline context to populate.
        parsed: Parsed JSON response from the LLM.
        result: CompletionResult with token usage information.
    """
    ctx.intent = parsed.get("intent", "")
    ctx.is_ambiguous = bool(parsed.get("is_ambiguous", False))
    ctx.token_usage += result.input_tokens + result.output_tokens
    ctx.model_call_count += 1
