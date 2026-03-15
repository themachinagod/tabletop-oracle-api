"""Unit tests for MessageService.

Tests session ownership validation logic and repository delegation
using mocked repositories. Covers happy path, session-not-found,
and ownership denial scenarios.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from tabletop_oracle.api.pagination import PaginationParams
from tabletop_oracle.auth.models import CurrentUser
from tabletop_oracle.errors.exceptions import NotFoundError
from tabletop_oracle.services.message_service import MessageService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

USER_ID = uuid.uuid4()
OTHER_USER_ID = uuid.uuid4()
SESSION_ID = uuid.uuid4()


def _make_user(*, user_id: uuid.UUID = USER_ID, role: str = "player") -> CurrentUser:
    """Build a CurrentUser for testing."""
    return CurrentUser(
        user_id=user_id,
        role=role,
        email="test@example.com",
        display_name="Test User",
        session_id="auth-session-1",
    )


@pytest.fixture
def mock_repo() -> AsyncMock:
    """Create a mock MessageRepository."""
    return AsyncMock()


@pytest.fixture
def service(mock_repo: AsyncMock) -> MessageService:
    """Create a MessageService with a mocked repository."""
    return MessageService(mock_repo)


@pytest.fixture
def pagination() -> PaginationParams:
    """Default pagination params for testing."""
    return PaginationParams(page=1, page_size=25)


# ---------------------------------------------------------------------------
# Tests: list_messages
# ---------------------------------------------------------------------------


class TestListMessages:
    """Tests for MessageService.list_messages."""

    @pytest.mark.asyncio
    async def test_returns_messages_for_session_owner(
        self,
        service: MessageService,
        mock_repo: AsyncMock,
        pagination: PaginationParams,
    ) -> None:
        """list_messages returns messages when user owns the session."""
        session = MagicMock()
        session.user_id = USER_ID
        mock_repo.get_session.return_value = session

        mock_messages = [MagicMock(), MagicMock()]
        mock_repo.list_by_session.return_value = (mock_messages, 2)

        user = _make_user()
        messages, total = await service.list_messages(SESSION_ID, user, pagination)

        assert total == 2
        assert len(messages) == 2
        mock_repo.get_session.assert_awaited_once_with(SESSION_ID)
        mock_repo.list_by_session.assert_awaited_once_with(SESSION_ID, pagination, order="asc")

    @pytest.mark.asyncio
    async def test_raises_not_found_when_session_missing(
        self,
        service: MessageService,
        mock_repo: AsyncMock,
        pagination: PaginationParams,
    ) -> None:
        """list_messages raises NotFoundError when session does not exist."""
        mock_repo.get_session.return_value = None
        user = _make_user()

        with pytest.raises(NotFoundError, match="Session"):
            await service.list_messages(SESSION_ID, user, pagination)

    @pytest.mark.asyncio
    async def test_raises_not_found_when_user_not_owner(
        self,
        service: MessageService,
        mock_repo: AsyncMock,
        pagination: PaginationParams,
    ) -> None:
        """list_messages raises NotFoundError when user does not own session."""
        session = MagicMock()
        session.user_id = OTHER_USER_ID
        mock_repo.get_session.return_value = session

        user = _make_user(user_id=USER_ID, role="player")

        with pytest.raises(NotFoundError, match="Session"):
            await service.list_messages(SESSION_ID, user, pagination)

    @pytest.mark.asyncio
    async def test_curator_bypasses_ownership_check(
        self,
        service: MessageService,
        mock_repo: AsyncMock,
        pagination: PaginationParams,
    ) -> None:
        """list_messages allows curator to access any session."""
        session = MagicMock()
        session.user_id = OTHER_USER_ID
        mock_repo.get_session.return_value = session
        mock_repo.list_by_session.return_value = ([], 0)

        curator = _make_user(user_id=USER_ID, role="curator")
        _messages, total = await service.list_messages(SESSION_ID, curator, pagination)

        assert total == 0

    @pytest.mark.asyncio
    async def test_passes_order_to_repository(
        self,
        service: MessageService,
        mock_repo: AsyncMock,
        pagination: PaginationParams,
    ) -> None:
        """list_messages forwards the order parameter to the repository."""
        session = MagicMock()
        session.user_id = USER_ID
        mock_repo.get_session.return_value = session
        mock_repo.list_by_session.return_value = ([], 0)

        user = _make_user()
        await service.list_messages(SESSION_ID, user, pagination, order="desc")

        mock_repo.list_by_session.assert_awaited_once_with(SESSION_ID, pagination, order="desc")
