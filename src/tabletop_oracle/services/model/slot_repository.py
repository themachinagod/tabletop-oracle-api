"""Data access for model slot configuration.

Queries the ``model_slots`` table by capability. No caching — every
call issues a fresh database read so admin changes take effect
immediately (AC-702).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select

from tabletop_oracle.models.model_slot import ModelSlot

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from tabletop_oracle.models.enums import ModelCapability


class ModelSlotRepository:
    """Repository for reading model slot configuration from the database.

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
