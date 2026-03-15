"""Unit tests for MessageRepository cancellation methods.

Tests get_message, set_cancelled, and is_cancelled using mocked
SQLAlchemy async sessions.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from tabletop_oracle.repositories.message_repository import MessageRepository

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MESSAGE_ID = uuid.uuid4()


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create a mock SQLAlchemy async session."""
    return AsyncMock()


@pytest.fixture
def repo(mock_session: AsyncMock) -> MessageRepository:
    """Create a MessageRepository with a mocked session."""
    return MessageRepository(mock_session)


# ---------------------------------------------------------------------------
# Tests: get_message
# ---------------------------------------------------------------------------


class TestGetMessage:
    """Tests for MessageRepository.get_message."""

    @pytest.mark.asyncio
    async def test_returns_message_when_found(
        self,
        repo: MessageRepository,
        mock_session: AsyncMock,
    ) -> None:
        """get_message returns the message when it exists."""
        from tabletop_oracle.models.message import Message

        expected = MagicMock(spec=Message)
        mock_session.get.return_value = expected

        result = await repo.get_message(MESSAGE_ID)

        assert result is expected
        mock_session.get.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(
        self,
        repo: MessageRepository,
        mock_session: AsyncMock,
    ) -> None:
        """get_message returns None when message does not exist."""
        mock_session.get.return_value = None

        result = await repo.get_message(MESSAGE_ID)

        assert result is None


# ---------------------------------------------------------------------------
# Tests: set_cancelled
# ---------------------------------------------------------------------------


class TestSetCancelled:
    """Tests for MessageRepository.set_cancelled."""

    @pytest.mark.asyncio
    async def test_sets_cancelled_at_and_flushes(
        self,
        repo: MessageRepository,
        mock_session: AsyncMock,
    ) -> None:
        """set_cancelled sets the timestamp and flushes the session."""
        message = MagicMock()
        message.cancelled_at = None
        now = datetime.now(UTC)

        result = await repo.set_cancelled(message, now)

        assert message.cancelled_at == now
        mock_session.flush.assert_awaited_once()
        assert result is message


# ---------------------------------------------------------------------------
# Tests: is_cancelled
# ---------------------------------------------------------------------------


class TestIsCancelled:
    """Tests for MessageRepository.is_cancelled."""

    @pytest.mark.asyncio
    async def test_returns_true_when_cancelled(
        self,
        repo: MessageRepository,
        mock_session: AsyncMock,
    ) -> None:
        """is_cancelled returns True when cancelled_at is not None."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = datetime.now(UTC)
        mock_session.execute.return_value = mock_result

        result = await repo.is_cancelled(MESSAGE_ID)

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_not_cancelled(
        self,
        repo: MessageRepository,
        mock_session: AsyncMock,
    ) -> None:
        """is_cancelled returns False when cancelled_at is None."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repo.is_cancelled(MESSAGE_ID)

        assert result is False
