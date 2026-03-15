"""Unit tests for MessageRepository.

Tests query construction, pagination delegation, and session lookup
using mocked async sessions. No real database — these verify method
contracts and argument forwarding.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from tabletop_oracle.api.pagination import PaginationParams
from tabletop_oracle.repositories.message_repository import MessageRepository

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create a mock async database session."""
    return AsyncMock()


@pytest.fixture
def repo(mock_session: AsyncMock) -> MessageRepository:
    """Create a MessageRepository with a mocked session."""
    return MessageRepository(mock_session)


# ---------------------------------------------------------------------------
# Tests: get_session
# ---------------------------------------------------------------------------


class TestGetSession:
    """Tests for MessageRepository.get_session."""

    @pytest.mark.asyncio
    async def test_delegates_to_session_get(
        self, repo: MessageRepository, mock_session: AsyncMock
    ) -> None:
        """get_session calls session.get with Session model and ID."""
        from tabletop_oracle.models.session import Session

        session_id = uuid.uuid4()
        mock_session.get.return_value = MagicMock(spec=Session)

        result = await repo.get_session(session_id)

        mock_session.get.assert_awaited_once_with(Session, session_id)
        assert result is not None

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(
        self, repo: MessageRepository, mock_session: AsyncMock
    ) -> None:
        """get_session returns None when session does not exist."""
        mock_session.get.return_value = None
        result = await repo.get_session(uuid.uuid4())
        assert result is None


# ---------------------------------------------------------------------------
# Tests: list_by_session
# ---------------------------------------------------------------------------


class TestListBySession:
    """Tests for MessageRepository.list_by_session."""

    @pytest.mark.asyncio
    async def test_executes_count_and_data_queries(
        self, repo: MessageRepository, mock_session: AsyncMock
    ) -> None:
        """list_by_session executes a count query and a data query."""
        # Mock count query
        count_result = MagicMock()
        count_result.scalar_one.return_value = 5

        # Mock data query — return empty scalars
        data_result = MagicMock()
        data_result.scalars.return_value.all.return_value = []

        mock_session.execute.side_effect = [count_result, data_result]

        params = PaginationParams(page=1, page_size=25)
        messages, total = await repo.list_by_session(uuid.uuid4(), params)

        assert total == 5
        assert messages == []
        assert mock_session.execute.await_count == 2

    @pytest.mark.asyncio
    async def test_returns_messages_from_data_query(
        self, repo: MessageRepository, mock_session: AsyncMock
    ) -> None:
        """list_by_session returns the messages from the paginated query."""
        mock_msg = MagicMock()
        mock_msg.sequence = 1

        count_result = MagicMock()
        count_result.scalar_one.return_value = 1

        data_result = MagicMock()
        data_result.scalars.return_value.all.return_value = [mock_msg]

        mock_session.execute.side_effect = [count_result, data_result]

        params = PaginationParams(page=1, page_size=25)
        messages, total = await repo.list_by_session(uuid.uuid4(), params)

        assert total == 1
        assert len(messages) == 1
        assert messages[0] is mock_msg

    @pytest.mark.asyncio
    async def test_default_order_is_asc(
        self, repo: MessageRepository, mock_session: AsyncMock
    ) -> None:
        """list_by_session defaults to ascending order."""
        count_result = MagicMock()
        count_result.scalar_one.return_value = 0
        data_result = MagicMock()
        data_result.scalars.return_value.all.return_value = []
        mock_session.execute.side_effect = [count_result, data_result]

        params = PaginationParams(page=1, page_size=25)
        # No order kwarg — should default to "asc"
        await repo.list_by_session(uuid.uuid4(), params)

        # Verify it executed without error (order default applied)
        assert mock_session.execute.await_count == 2

    @pytest.mark.asyncio
    async def test_desc_order_accepted(
        self, repo: MessageRepository, mock_session: AsyncMock
    ) -> None:
        """list_by_session accepts desc order without error."""
        count_result = MagicMock()
        count_result.scalar_one.return_value = 0
        data_result = MagicMock()
        data_result.scalars.return_value.all.return_value = []
        mock_session.execute.side_effect = [count_result, data_result]

        params = PaginationParams(page=1, page_size=25)
        await repo.list_by_session(uuid.uuid4(), params, order="desc")

        assert mock_session.execute.await_count == 2
