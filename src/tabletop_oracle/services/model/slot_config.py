"""Immutable model slot configuration dataclass.

Represents a point-in-time snapshot of the model configuration for a
given AI capability. Frozen to prevent accidental mutation after retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tabletop_oracle.models.enums import ModelCapability


@dataclass(frozen=True)
class ModelSlotConfig:
    """Immutable snapshot of a model slot configuration.

    Attributes:
        capability: The AI capability this slot serves.
        provider: Model provider identifier (e.g. 'openai', 'anthropic').
        model_id: Provider-specific model identifier.
        max_tokens_per_call: Maximum tokens per API call, or None if unlimited.
        temperature: Sampling temperature, or None for provider default.
        fallback_provider: Fallback provider if primary fails, or None.
        fallback_model_id: Fallback model identifier, or None.
    """

    capability: ModelCapability
    provider: str
    model_id: str
    max_tokens_per_call: int | None
    temperature: float | None
    fallback_provider: str | None
    fallback_model_id: str | None
