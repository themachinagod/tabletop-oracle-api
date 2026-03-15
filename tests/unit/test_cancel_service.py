"""Unit tests for MessageService.cancel_message.

Tests session ownership validation, message existence checks,
already-complete and already-cancelled scenarios, and successful
cancellation using mocked repositories.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from tabletop_oracle.auth.models import CurrentUser
from tabletop_oracle.errors.exceptions import NotFoundError
from tabletop_oracle.services.message_service import MessageService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

USER_ID = uuid.uuid4()
OTHER_USER_ID = uuid.uuid4()
SESSION_ID = uuid.uuid4()
MESSAGE_ID = uuid.uuid4()


def _make_user(*, user_id: uuid.UUID = USER_ID, role: str = "player") -> CurrentUser:
    """Build a CurrentUser for testing."""
    return CurrentUser(
        user_id=user_id,
        role=role,
        email="test@example.com",
        display_name="Test User",
        session_id="auth-session-1",
    )


def _make_session(*, user_id: uuid.UUID = USER_ID) -> MagicMock:
    """Build a mock game session."""
    session = MagicMock()
    session.user_id = user_id
    return session


def _make_message(
    *,
    session_id: uuid.UUID = SESSION_ID,
    confidence_score: float | None = None,
    cancelled_at: datetime | None = None,
) -> MagicMock:
    """Build a mock Message entity."""
    msg = MagicMock()
    msg.id = MESSAGE_ID
    msg.session_id = session_id
    msg.confidence_score = confidence_score
    msg.cancelled_at = cancelled_at
    return msg


@pytest.fixture
def mock_repo() -> AsyncMock:
    """Create a mock MessageRepository."""
    return AsyncMock()


@pytest.fixture
def service(mock_repo: AsyncMock) -> MessageService:
    """Create a MessageService with a mocked repository."""
    return MessageService(mock_repo)


# ---------------------------------------------------------------------------
# Tests: cancel_message
# ---------------------------------------------------------------------------


class TestCancelMessage:
    """Tests for MessageService.cancel_message."""

    @pytest.mark.asyncio
    async def test_cancel_active_message_returns_true(
        self,
        service: MessageService,
        mock_repo: AsyncMock,
    ) -> None:
        """cancel_message sets cancelled_at and returns (message, True) for active query."""
        mock_repo.get_session.return_value = _make_session()
        message = _make_message()
        mock_repo.get_message.return_value = message
        mock_repo.set_cancelled.return_value = message

        user = _make_user()
        _result_msg, was_cancelled = await service.cancel_message(
            SESSION_ID,
            MESSAGE_ID,
            user,
        )

        assert was_cancelled is True
        mock_repo.set_cancelled.assert_awaited_once()
        # Verify the timestamp argument is recent
        call_args = mock_repo.set_cancelled.call_args
        assert call_args[0][0] is message
        assert isinstance(call_args[0][1], datetime)

    @pytest.mark.asyncio
    async def test_already_complete_returns_false(
        self,
        service: MessageService,
        mock_repo: AsyncMock,
    ) -> None:
        """cancel_message returns (message, False) when message has confidence_score."""
        mock_repo.get_session.return_value = _make_session()
        message = _make_message(confidence_score=0.85)
        mock_repo.get_message.return_value = message

        user = _make_user()
        _result_msg, was_cancelled = await service.cancel_message(
            SESSION_ID,
            MESSAGE_ID,
            user,
        )

        assert was_cancelled is False
        mock_repo.set_cancelled.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_already_cancelled_returns_false(
        self,
        service: MessageService,
        mock_repo: AsyncMock,
    ) -> None:
        """cancel_message returns (message, False) when already cancelled."""
        mock_repo.get_session.return_value = _make_session()
        message = _make_message(cancelled_at=datetime.now(UTC))
        mock_repo.get_message.return_value = message

        user = _make_user()
        _result_msg, was_cancelled = await service.cancel_message(
            SESSION_ID,
            MESSAGE_ID,
            user,
        )

        assert was_cancelled is False
        mock_repo.set_cancelled.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_session_not_found_raises(
        self,
        service: MessageService,
        mock_repo: AsyncMock,
    ) -> None:
        """cancel_message raises NotFoundError when session does not exist."""
        mock_repo.get_session.return_value = None

        user = _make_user()
        with pytest.raises(NotFoundError, match="Session"):
            await service.cancel_message(SESSION_ID, MESSAGE_ID, user)

    @pytest.mark.asyncio
    async def test_message_not_found_raises(
        self,
        service: MessageService,
        mock_repo: AsyncMock,
    ) -> None:
        """cancel_message raises NotFoundError when message does not exist."""
        mock_repo.get_session.return_value = _make_session()
        mock_repo.get_message.return_value = None

        user = _make_user()
        with pytest.raises(NotFoundError, match="Message"):
            await service.cancel_message(SESSION_ID, MESSAGE_ID, user)

    @pytest.mark.asyncio
    async def test_message_wrong_session_raises(
        self,
        service: MessageService,
        mock_repo: AsyncMock,
    ) -> None:
        """cancel_message raises NotFoundError when message belongs to a different session."""
        mock_repo.get_session.return_value = _make_session()
        other_session_id = uuid.uuid4()
        message = _make_message(session_id=other_session_id)
        mock_repo.get_message.return_value = message

        user = _make_user()
        with pytest.raises(NotFoundError, match="Message"):
            await service.cancel_message(SESSION_ID, MESSAGE_ID, user)

    @pytest.mark.asyncio
    async def test_non_owner_raises_not_found(
        self,
        service: MessageService,
        mock_repo: AsyncMock,
    ) -> None:
        """cancel_message raises NotFoundError when user does not own the session."""
        mock_repo.get_session.return_value = _make_session(user_id=OTHER_USER_ID)

        user = _make_user(user_id=USER_ID, role="player")
        with pytest.raises(NotFoundError, match="Session"):
            await service.cancel_message(SESSION_ID, MESSAGE_ID, user)

    @pytest.mark.asyncio
    async def test_curator_bypasses_ownership(
        self,
        service: MessageService,
        mock_repo: AsyncMock,
    ) -> None:
        """cancel_message allows curator to cancel any session's message."""
        mock_repo.get_session.return_value = _make_session(user_id=OTHER_USER_ID)
        message = _make_message()
        mock_repo.get_message.return_value = message
        mock_repo.set_cancelled.return_value = message

        curator = _make_user(user_id=USER_ID, role="curator")
        _msg, was_cancelled = await service.cancel_message(
            SESSION_ID,
            MESSAGE_ID,
            curator,
        )

        assert was_cancelled is True
