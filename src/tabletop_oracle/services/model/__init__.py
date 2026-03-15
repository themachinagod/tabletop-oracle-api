"""Model services — slot configuration, LLM client, and supporting protocols.

Provides runtime AI model configuration, a provider-agnostic LLM client
(via litellm), and protocol interfaces for token usage tracking and
guardrail enforcement.
"""

from tabletop_oracle.services.model.client import ModelClient
from tabletop_oracle.services.model.guardrail import GuardrailService, GuardrailServiceProtocol
from tabletop_oracle.services.model.slot_config import ModelSlotConfig
from tabletop_oracle.services.model.slot_repository import ModelSlotRepository
from tabletop_oracle.services.model.slot_service import ModelSlotService
from tabletop_oracle.services.model.token_usage import (
    TokenAttribution,
    TokenUsageService,
    TokenUsageServiceProtocol,
)
from tabletop_oracle.services.model.types import CompletionResult, GuardrailCheckResult

__all__ = [
    "CompletionResult",
    "GuardrailCheckResult",
    "GuardrailService",
    "GuardrailServiceProtocol",
    "ModelClient",
    "ModelSlotConfig",
    "ModelSlotRepository",
    "ModelSlotService",
    "TokenAttribution",
    "TokenUsageService",
    "TokenUsageServiceProtocol",
]
