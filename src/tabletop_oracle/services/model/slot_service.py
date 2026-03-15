"""Business logic for runtime AI model configuration.

Reads from the ``model_slots`` table via the repository layer. No
caching — every call reads fresh configuration so that admin changes
take effect immediately (AC-702).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tabletop_oracle.errors.exceptions import ConfigurationError
from tabletop_oracle.services.model.slot_config import ModelSlotConfig

if TYPE_CHECKING:
    from tabletop_oracle.models.enums import ModelCapability
    from tabletop_oracle.services.model.slot_repository import ModelSlotRepository


class ModelSlotService:
    """Provides runtime model configuration from the database.

    Reads from the model_slots table. No caching -- every call reads
    fresh configuration so that admin changes take effect immediately
    (AC-702).

    Args:
        repository: Repository for model slot data access.
    """

    def __init__(self, repository: ModelSlotRepository) -> None:
        self._repository = repository

    async def get_slot(self, capability: ModelCapability) -> ModelSlotConfig:
        """Get the current model configuration for a capability.

        Args:
            capability: The AI capability to retrieve configuration for.

        Returns:
            ModelSlotConfig with provider, model_id, temperature,
            max_tokens_per_call, and optional fallback config.

        Raises:
            ConfigurationError: If no slot is configured for the capability.
        """
        slot = await self._repository.get_by_capability(capability)
        if slot is None:
            raise ConfigurationError(
                f"No model slot configured for capability '{capability.value}'"
            )

        return ModelSlotConfig(
            capability=slot.capability,
            provider=slot.provider,
            model_id=slot.model_id,
            max_tokens_per_call=slot.max_tokens_per_call,
            temperature=slot.temperature,
            fallback_provider=slot.fallback_provider,
            fallback_model_id=slot.fallback_model_id,
        )
