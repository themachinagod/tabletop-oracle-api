"""Data access for model slot configuration.

Queries the ``model_slots`` table by capability. No caching — every
call issues a fresh database read so admin changes take effect
immediately (AC-702).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from tabletop_oracle.models.model_slot import ModelSlot

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from tabletop_oracle.models.enums import ModelCapability


class ModelSlotRepository:
    """Repository for model slot configuration data access.

    Supports read-by-capability, list-all, and partial update operations.
    No caching -- every call reads fresh from the database (AC-702).

    Args:
        session: SQLAlchemy async session for database operations.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_capability(self, capability: ModelCapability) -> ModelSlot | None:
        """Fetch the model slot row for a given capability.

        Args:
            capability: The AI capability to look up.

        Returns:
            The ModelSlot ORM entity, or None if no slot is configured.
        """
        stmt = select(ModelSlot).where(ModelSlot.capability == capability)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_all(self) -> list[ModelSlot]:
        """Fetch all model slot rows ordered by capability.

        Returns:
            List of all ModelSlot ORM entities.
        """
        stmt = select(ModelSlot).order_by(ModelSlot.capability)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def update(self, slot: ModelSlot, fields: dict[str, Any]) -> ModelSlot:
        """Apply partial updates to a model slot.

        Args:
            slot: The ModelSlot ORM entity to update.
            fields: Mapping of field names to new values.

        Returns:
            The updated ModelSlot entity (flushed, not yet committed).
        """
        for key, value in fields.items():
            setattr(slot, key, value)
        await self._session.flush()
        await self._session.refresh(slot)
        return slot
