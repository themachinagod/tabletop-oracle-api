"""Unit tests for POST /api/v1/sessions/{session_id}/messages.

Tests endpoint through ASGI transport with bypass_auth enabled.
Creates a fresh app for each test to ensure routes are registered.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from tabletop_oracle.errors.exceptions import (
    ConflictError,
    ContentTooLargeError,
    NotFoundError,
    UnprocessableEntityError,
)
from tabletop_oracle.models.enums import ContextAttachmentType, MessageType
from tabletop_oracle.services.message_service import MessageService

_SESSION_ID = uuid.UUID("33333333-3333-3333-3333-333333333333")
_MESSAGE_ID = uuid.UUID("44444444-4444-4444-4444-444444444444")
_ATT_ID = uuid.UUID("55555555-5555-5555-5555-555555555555")

_ENDPOINT = f"/api/v1/sessions/{_SESSION_ID}/messages"


def _make_message_orm() -> MagicMock:
    """Build a mock Message ORM object."""
    msg = MagicMock()
    msg.id = _MESSAGE_ID
    msg.session_id = _SESSION_ID
    msg.type = MessageType.USER_QUESTION
    msg.content = "How does combat work?"
    msg.sequence = 1
    msg.created_at = datetime(2026, 1, 1, tzinfo=UTC)
    msg.in_reply_to_id = None
    msg.confidence_score = None
    msg.processing_duration_ms = None

    att = MagicMock()
    att.id = _ATT_ID
    att.type = ContextAttachmentType.TEXT
    att.content = "Extra context"
    att.file_path = None
    att.file_name = None
    att.file_size = None
    att.processed_description = None

    msg.citations = []
    msg.context_attachments = [att]
    return msg


class TestSubmitMessageEndpoint:
    """Tests for POST /api/v1/sessions/{session_id}/messages."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> Any:
        """Override DB dependency."""
        from tabletop_oracle.api.deps import get_db
        from tabletop_oracle.main import app

        async def _fake_db() -> Any:
            yield MagicMock()

        app.dependency_overrides[get_db] = _fake_db
        yield
        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_successful_json_submission_returns_201(self) -> None:
        """Valid JSON returns 201 with message data."""
        from tabletop_oracle.main import app

        with (
            patch("tabletop_oracle.auth.middleware.settings") as ms,
            patch.object(
                MessageService,
                "submit_message",
                new_callable=AsyncMock,
                return_value=_make_message_orm(),
            ),
        ):
            ms.bypass_auth = True
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                r = await c.post(_ENDPOINT, json={"content": "How does combat work?"})

        assert r.status_code == 201
        data = r.json()["data"]
        assert data["type"] == "user_question"
        assert data["sequence"] == 1
        assert len(data["context_attachments"]) == 1

    @pytest.mark.asyncio
    async def test_session_not_found_returns_404(self) -> None:
        """Non-existent session returns 404."""
        from tabletop_oracle.main import app

        with (
            patch("tabletop_oracle.auth.middleware.settings") as ms,
            patch.object(
                MessageService,
                "submit_message",
                new_callable=AsyncMock,
                side_effect=NotFoundError("Session", str(_SESSION_ID)),
            ),
        ):
            ms.bypass_auth = True
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                r = await c.post(_ENDPOINT, json={"content": "Q"})

        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_archived_session_returns_409(self) -> None:
        """Archived session returns 409."""
        from tabletop_oracle.main import app

        with (
            patch("tabletop_oracle.auth.middleware.settings") as ms,
            patch.object(
                MessageService,
                "submit_message",
                new_callable=AsyncMock,
                side_effect=ConflictError("Session is archived"),
            ),
        ):
            ms.bypass_auth = True
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                r = await c.post(_ENDPOINT, json={"content": "Q"})

        assert r.status_code == 409

    @pytest.mark.asyncio
    async def test_unauthenticated_returns_401(self) -> None:
        """No auth returns 401."""
        from tabletop_oracle.main import app

        with patch("tabletop_oracle.auth.middleware.settings") as ms:
            ms.bypass_auth = False
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                r = await c.post(_ENDPOINT, json={"content": "Q"})

        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_empty_content_returns_422(self) -> None:
        """Empty content returns 422."""
        from tabletop_oracle.main import app

        with (
            patch("tabletop_oracle.auth.middleware.settings") as ms,
            patch.object(
                MessageService,
                "submit_message",
                new_callable=AsyncMock,
                side_effect=UnprocessableEntityError("Message content must not be empty"),
            ),
        ):
            ms.bypass_auth = True
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                r = await c.post(_ENDPOINT, json={"content": "   "})

        assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_image_too_large_returns_413(self) -> None:
        """Oversized image returns 413."""
        from tabletop_oracle.main import app

        with (
            patch("tabletop_oracle.auth.middleware.settings") as ms,
            patch.object(
                MessageService,
                "submit_message",
                new_callable=AsyncMock,
                side_effect=ContentTooLargeError("Image exceeds 10MB"),
            ),
        ):
            ms.bypass_auth = True
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                r = await c.post(
                    _ENDPOINT,
                    json={
                        "content": "Check",
                        "context_attachments": [
                            {"type": "image", "file_name": "big.jpg"},
                        ],
                    },
                )

        assert r.status_code == 413

    @pytest.mark.asyncio
    async def test_response_has_envelope_structure(self) -> None:
        """Response follows DataEnvelope structure."""
        from tabletop_oracle.main import app

        with (
            patch("tabletop_oracle.auth.middleware.settings") as ms,
            patch.object(
                MessageService,
                "submit_message",
                new_callable=AsyncMock,
                return_value=_make_message_orm(),
            ),
        ):
            ms.bypass_auth = True
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                r = await c.post(_ENDPOINT, json={"content": "Q"})

        body = r.json()
        assert "data" in body
        assert "meta" in body
        assert "request_id" in body["meta"]

    @pytest.mark.asyncio
    async def test_service_submit_called(self) -> None:
        """Service submit_message is called on success."""
        from tabletop_oracle.main import app

        with (
            patch("tabletop_oracle.auth.middleware.settings") as ms,
            patch.object(
                MessageService,
                "submit_message",
                new_callable=AsyncMock,
                return_value=_make_message_orm(),
            ) as mock_submit,
        ):
            ms.bypass_auth = True
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                await c.post(_ENDPOINT, json={"content": "Q"})

        mock_submit.assert_awaited_once()
