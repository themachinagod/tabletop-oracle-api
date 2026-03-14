"""Unit tests for the base repository pattern."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from tabletop_oracle.repositories.base import BaseRepository


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create a mock async session."""
    session = AsyncMock()
    return session


@pytest.fixture
def mock_model_class() -> type:
    """Create a mock model class."""
    return MagicMock()


@pytest.mark.asyncio
async def test_get_by_id(mock_session: AsyncMock, mock_model_class: type) -> None:
    """get_by_id delegates to session.get."""
    import uuid

    repo = BaseRepository(mock_session, mock_model_class)
    entity_id = uuid.uuid4()
    await repo.get_by_id(entity_id)
    mock_session.get.assert_awaited_once_with(mock_model_class, entity_id)


@pytest.mark.asyncio
async def test_create(mock_session: AsyncMock, mock_model_class: type) -> None:
    """create adds entity to session and flushes."""
    repo = BaseRepository(mock_session, mock_model_class)
    entity = MagicMock()
    result = await repo.create(entity)
    mock_session.add.assert_called_once_with(entity)
    mock_session.flush.assert_awaited_once()
    assert result is entity


@pytest.mark.asyncio
async def test_delete(mock_session: AsyncMock, mock_model_class: type) -> None:
    """delete removes entity from session."""
    repo = BaseRepository(mock_session, mock_model_class)
    entity = MagicMock()
    await repo.delete(entity)
    mock_session.delete.assert_awaited_once_with(entity)


@pytest.mark.asyncio
async def test_list_all(mock_session: AsyncMock, mock_model_class: type) -> None:
    """list_all executes a paginated query."""
    repo = BaseRepository(mock_session, mock_model_class)
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = ["item1", "item2"]
    mock_session.execute.return_value = mock_result
    result = await repo.list_all(offset=0, limit=10)
    assert result == ["item1", "item2"]
    mock_session.execute.assert_awaited_once()
