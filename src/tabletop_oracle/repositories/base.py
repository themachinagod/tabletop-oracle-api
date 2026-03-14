"""Base repository with common CRUD operations."""

import uuid
from typing import Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tabletop_oracle.models.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    """Generic repository providing standard CRUD operations."""

    def __init__(self, session: AsyncSession, model_class: type[ModelT]) -> None:
        self._session = session
        self._model_class = model_class

    async def get_by_id(self, entity_id: uuid.UUID) -> ModelT | None:
        """Fetch a single entity by primary key."""
        return await self._session.get(self._model_class, entity_id)

    async def list_all(self, *, offset: int = 0, limit: int = 20) -> list[ModelT]:
        """Fetch a paginated list of entities."""
        stmt = select(self._model_class).offset(offset).limit(limit)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def create(self, entity: ModelT) -> ModelT:
        """Persist a new entity."""
        self._session.add(entity)
        await self._session.flush()
        return entity

    async def delete(self, entity: ModelT) -> None:
        """Remove an entity."""
        await self._session.delete(entity)
