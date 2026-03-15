"""Unit tests for MessageService.submit_message business logic."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from tabletop_oracle.auth.models import CurrentUser
from tabletop_oracle.errors.exceptions import (
    ConflictError,
    ContentTooLargeError,
    NotFoundError,
    UnprocessableEntityError,
)
from tabletop_oracle.models.enums import MessageType, SessionStatus
from tabletop_oracle.schemas.message import (
    ContextAttachmentRequest,
    MessageSubmitRequest,
)
from tabletop_oracle.services.message_service import MAX_IMAGE_SIZE_BYTES, MessageService

_USER_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
_OTHER_USER_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")
_SESSION_ID = uuid.UUID("33333333-3333-3333-3333-333333333333")
_MESSAGE_ID = uuid.UUID("44444444-4444-4444-4444-444444444444")


def _make_user(user_id: uuid.UUID = _USER_ID, role: str = "player") -> CurrentUser:
    """Build a CurrentUser fixture."""
    return CurrentUser(
        user_id=user_id,
        role=role,
        email="t@t.com",
        display_name="T",
        session_id="s",
    )


def _make_session(
    user_id: uuid.UUID = _USER_ID,
    status: SessionStatus = SessionStatus.ACTIVE,
) -> MagicMock:
    """Build a mock Session ORM object."""
    s = MagicMock()
    s.id = _SESSION_ID
    s.user_id = user_id
    s.status = status
    return s


def _make_message_mock(sequence: int = 1) -> MagicMock:
    """Build a mock Message ORM object."""
    msg = MagicMock()
    msg.id = _MESSAGE_ID
    msg.session_id = _SESSION_ID
    msg.type = MessageType.USER_QUESTION
    msg.content = "Test"
    msg.sequence = sequence
    msg.created_at = datetime(2026, 1, 1, tzinfo=UTC)
    msg.context_attachments = []
    return msg


def _build_service(
    session_mock: object | None = None,
    sequence: int = 1,
) -> MessageService:
    """Build a MessageService with mocked dependencies."""
    repo = MagicMock()
    repo.get_session = AsyncMock(return_value=session_mock)
    repo.get_next_sequence = AsyncMock(return_value=sequence)
    msg = _make_message_mock(sequence)
    repo.create_message = AsyncMock(return_value=msg)
    repo.create_attachments = AsyncMock(return_value=[])
    return MessageService(message_repo=repo)


class TestSubmitMessage:
    """Tests for MessageService.submit_message."""

    @pytest.mark.asyncio
    async def test_successful_submission(self) -> None:
        """Happy path returns persisted message."""
        svc = _build_service(session_mock=_make_session(), sequence=5)
        result = await svc.submit_message(
            _SESSION_ID,
            MessageSubmitRequest(content="Q"),
            _make_user(),
        )
        assert result.sequence == 5

    @pytest.mark.asyncio
    async def test_session_not_found_raises_404(self) -> None:
        """Missing session raises NotFoundError."""
        svc = _build_service(session_mock=None)
        with pytest.raises(NotFoundError):
            await svc.submit_message(
                _SESSION_ID,
                MessageSubmitRequest(content="Q"),
                _make_user(),
            )

    @pytest.mark.asyncio
    async def test_session_not_owned_raises_404(self) -> None:
        """Session owned by another user raises NotFoundError."""
        svc = _build_service(session_mock=_make_session(user_id=_OTHER_USER_ID))
        with pytest.raises(NotFoundError):
            await svc.submit_message(
                _SESSION_ID,
                MessageSubmitRequest(content="Q"),
                _make_user(),
            )

    @pytest.mark.asyncio
    async def test_archived_session_raises_409(self) -> None:
        """Archived session raises ConflictError."""
        svc = _build_service(session_mock=_make_session(status=SessionStatus.ARCHIVED))
        with pytest.raises(ConflictError, match="archived"):
            await svc.submit_message(
                _SESSION_ID,
                MessageSubmitRequest(content="Q"),
                _make_user(),
            )

    @pytest.mark.asyncio
    async def test_whitespace_content_raises_422(self) -> None:
        """Whitespace-only content raises UnprocessableEntityError."""
        svc = _build_service(session_mock=_make_session())
        with pytest.raises(UnprocessableEntityError, match="empty"):
            await svc.submit_message(
                _SESSION_ID,
                MessageSubmitRequest(content="   "),
                _make_user(),
            )

    @pytest.mark.asyncio
    async def test_image_too_large_raises_413(self) -> None:
        """Image exceeding 10MB raises ContentTooLargeError."""
        svc = _build_service(session_mock=_make_session())
        req = MessageSubmitRequest(
            content="Check",
            context_attachments=[ContextAttachmentRequest(type="image", file_name="big.jpg")],
        )
        with pytest.raises(ContentTooLargeError):
            await svc.submit_message(
                _SESSION_ID,
                req,
                _make_user(),
                image_files={"big.jpg": (b"x" * (MAX_IMAGE_SIZE_BYTES + 1), "image/jpeg")},
            )

    @pytest.mark.asyncio
    async def test_attachments_persisted(self) -> None:
        """Text attachments passed to repo."""
        svc = _build_service(session_mock=_make_session())
        req = MessageSubmitRequest(
            content="Q",
            context_attachments=[ContextAttachmentRequest(type="text", content="ctx")],
        )
        await svc.submit_message(_SESSION_ID, req, _make_user())
        svc._repo.create_attachments.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_curator_bypasses_ownership(self) -> None:
        """Curator can submit to any session."""
        svc = _build_service(session_mock=_make_session(user_id=_OTHER_USER_ID))
        result = await svc.submit_message(
            _SESSION_ID,
            MessageSubmitRequest(content="Q"),
            _make_user(role="curator"),
        )
        assert result.id == _MESSAGE_ID
