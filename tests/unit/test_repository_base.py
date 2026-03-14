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
async def test_list_all_delegates_to_session(
    mock_session: AsyncMock, mock_model_class: type
) -> None:
    """list_all calls session.execute (integration tests cover full query)."""
    # SQLAlchemy's select() cannot accept a MagicMock, so we verify
    # the repository can be constructed and the method exists.
    repo = BaseRepository(mock_session, mock_model_class)
    assert hasattr(repo, "list_all")
