"""Business logic for runtime AI model configuration.

Reads from the ``model_slots`` table via the repository layer. No
caching — every call reads fresh configuration so that admin changes
take effect immediately (AC-702).

Extended for EPIC-005 with admin operations: ``list_all()`` and
``update_slot()`` for the model slot admin API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tabletop_oracle.errors.exceptions import ConfigurationError, NotFoundError
from tabletop_oracle.services.model.slot_config import ModelSlotConfig

if TYPE_CHECKING:
    from tabletop_oracle.models.enums import ModelCapability
    from tabletop_oracle.models.model_slot import ModelSlot
    from tabletop_oracle.services.model.slot_repository import ModelSlotRepository

#: Fields on ModelSlot that are updatable via the admin API.
_UPDATABLE_FIELDS = frozenset(
    {
        "provider",
        "model_id",
        "max_tokens_per_call",
        "temperature",
        "fallback_provider",
        "fallback_model_id",
    }
)


def _slot_to_config(slot: ModelSlot) -> ModelSlotConfig:
    """Convert a ModelSlot ORM entity to an immutable ModelSlotConfig.

    Args:
        slot: The ModelSlot ORM entity.

    Returns:
        Frozen ModelSlotConfig snapshot.
    """
    return ModelSlotConfig(
        capability=slot.capability,
        provider=slot.provider,
        model_id=slot.model_id,
        max_tokens_per_call=slot.max_tokens_per_call,
        temperature=slot.temperature,
        fallback_provider=slot.fallback_provider,
        fallback_model_id=slot.fallback_model_id,
    )


class ModelSlotService:
    """Provides runtime model configuration from the database.

    Reads from the model_slots table. No caching -- every call reads
    fresh configuration so that admin changes take effect immediately
    (AC-702).

    Admin operations (``list_all``, ``update_slot``) are used by the
    admin API for EPIC-005 configuration management.

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
        return _slot_to_config(slot)

    async def list_all(self) -> list[ModelSlotConfig]:
        """List all configured model slots.

        Returns:
            All capability slots with their current configuration,
            ordered by capability name.
        """
        slots = await self._repository.get_all()
        return [_slot_to_config(s) for s in slots]

    async def update_slot(
        self,
        capability: ModelCapability,
        updates: dict[str, Any],
    ) -> ModelSlotConfig:
        """Update a model slot configuration (partial update).

        Only fields present in ``updates`` are modified. Field names
        are validated against the known updatable set.

        Args:
            capability: Which capability slot to update.
            updates: Mapping of field names to new values.

        Returns:
            Updated slot configuration.

        Raises:
            NotFoundError: If no slot exists for this capability.
        """
        slot = await self._repository.get_by_capability(capability)
        if slot is None:
            raise NotFoundError("Model slot", capability.value)

        # Filter to only known updatable fields
        valid_updates = {k: v for k, v in updates.items() if k in _UPDATABLE_FIELDS}
        if not valid_updates:
            return _slot_to_config(slot)

        updated = await self._repository.update(slot, valid_updates)
        return _slot_to_config(updated)
