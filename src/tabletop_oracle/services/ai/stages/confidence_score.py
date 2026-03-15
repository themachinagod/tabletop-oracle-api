"""Stage 7: Confidence scoring for the query pipeline.

Self-assesses answer confidence via a lightweight LLM call using the
``intent_analysis`` model slot (fast, cheap). Evaluates how well the
generated answer is supported by the knowledge graph sources.

Design reference: EPIC-004 design.md, Stage 7: Confidence Score.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from tabletop_oracle.models.enums import ModelCapability
from tabletop_oracle.services.ai.prompts.confidence import build_confidence_messages
from tabletop_oracle.streaming.payloads import StreamConfidencePayload, payload_to_sse_event

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from tabletop_oracle.services.ai.context import PipelineContext
    from tabletop_oracle.services.model.client import ModelClient
    from tabletop_oracle.services.model.token_usage import TokenAttribution
    from tabletop_oracle.streaming.events import SseEvent

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = {"score", "justification"}

# Label thresholds per design doc.
_HIGH_THRESHOLD = 0.8
_MODERATE_THRESHOLD = 0.5


def _score_to_label(score: float) -> str:
    """Map a confidence score to a human-readable label.

    Args:
        score: Confidence value between 0.0 and 1.0.

    Returns:
        'High', 'Moderate', or 'Low'.
    """
    if score >= _HIGH_THRESHOLD:
        return "High"
    if score >= _MODERATE_THRESHOLD:
        return "Moderate"
    return "Low"


class ConfidenceScoreStage:
    """Self-assesses answer confidence via prompted logic.

    Uses the ``intent_analysis`` model slot (fast, cheap) to evaluate
    confidence based on source quality, agreement, completeness, and
    inference level. Returns a float 0.0-1.0 with justification.

    Populates on ``PipelineContext``:
    - ``confidence_score``: Float between 0.0 and 1.0.
    - ``confidence_label``: Human-readable label (High/Moderate/Low).
    - ``token_usage``: Incremented by tokens consumed.
    - ``model_call_count``: Incremented by 1.

    Yields:
    - ``stream.confidence`` SSE event with score and label.
    """

    async def execute(
        self,
        ctx: PipelineContext,
        model_client: ModelClient,
    ) -> AsyncGenerator[SseEvent, None]:
        """Assess confidence in the generated answer.

        Calls the LLM with the confidence prompt, parses the JSON response,
        assigns a label, updates the pipeline context, logs the justification,
        and yields a ``stream.confidence`` SSE event.

        Args:
            ctx: Pipeline context with answer_text and retrieval results.
            model_client: Provider-agnostic model client for LLM calls.

        Yields:
            A single ``stream.confidence`` SSE event.

        Raises:
            ValueError: If the LLM response cannot be parsed as valid
                confidence scoring JSON.
        """
        messages = build_confidence_messages(ctx)
        attribution = _build_attribution(ctx)

        result = await model_client.complete(
            capability=ModelCapability.INTENT_ANALYSIS,
            messages=messages,
            attribution=attribution,
        )

        parsed = _parse_response(result.content)
        score = _clamp_score(parsed["score"])
        label = _score_to_label(score)

        ctx.confidence_score = score
        ctx.confidence_label = label
        ctx.token_usage += result.input_tokens + result.output_tokens
        ctx.model_call_count += 1

        logger.info(
            "Confidence scoring complete",
            extra={
                "message_id": str(ctx.user_message.id),
                "score": score,
                "label": label,
                "justification": parsed.get("justification", ""),
            },
        )

        yield payload_to_sse_event(StreamConfidencePayload(score=score, label=label))


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
    """Parse the LLM response as confidence scoring JSON.

    Handles responses that may be wrapped in markdown code fences.

    Args:
        content: Raw LLM response text.

    Returns:
        Parsed dict with score and justification fields.

    Raises:
        ValueError: If the response is not valid JSON or is missing
            required keys.
    """
    cleaned = content.strip()

    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [
            line
            for i, line in enumerate(lines)
            if not (i == 0 or (i == len(lines) - 1 and line.strip() == "```"))
        ]
        cleaned = "\n".join(lines)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Confidence scoring response is not valid JSON: {content[:200]}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            f"Confidence scoring response is not a JSON object: {type(parsed).__name__}"
        )

    missing = _REQUIRED_KEYS - set(parsed.keys())
    if missing:
        raise ValueError(f"Confidence scoring response missing required keys: {missing}")

    if not isinstance(parsed["score"], (int, float)):
        raise ValueError(
            f"Confidence score must be a number, got {type(parsed['score']).__name__}"
        )

    return parsed


def _clamp_score(raw_score: float) -> float:
    """Clamp a raw score to the valid 0.0-1.0 range.

    Defensive — the LLM should return a valid score, but we clamp
    to guarantee the contract.

    Args:
        raw_score: The score value from the LLM response.

    Returns:
        Score clamped to [0.0, 1.0].
    """
    return max(0.0, min(1.0, float(raw_score)))
