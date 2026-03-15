"""Data types for the model client layer.

Contains value objects for completion results and guardrail check results
used across the model client and guardrail services.

Note: ``TokenAttribution`` lives in ``token_usage.py`` alongside the
``TokenUsageService`` that consumes it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompletionResult:
    """Result of a non-streaming model completion call.

    Attributes:
        content: The generated text content.
        input_tokens: Number of input tokens consumed.
        output_tokens: Number of output tokens generated.
        model_used: The litellm model identifier actually used (may be fallback).
        is_fallback: Whether the fallback model was used.
    """

    content: str
    input_tokens: int
    output_tokens: int
    model_used: str
    is_fallback: bool


@dataclass(frozen=True)
class GuardrailCheckResult:
    """Result of a guardrail check.

    Attributes:
        allowed: Whether the operation is permitted.
        message: Human-readable denial message if not allowed.
        guardrail_type: Which guardrail was hit (for logging).
    """

    allowed: bool
    message: str | None = None
    guardrail_type: str | None = None
