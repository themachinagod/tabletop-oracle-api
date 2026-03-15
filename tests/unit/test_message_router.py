"""Unit tests for the message history API router.

Tests endpoint behaviour through the ASGI transport with bypass_auth
enabled (middleware injects a player user). Verifies F001 envelope
structure, pagination, ordering, and error handling.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from tabletop_oracle.main import app
from tabletop_oracle.models.message import Message
from tabletop_oracle.services.message_service import MessageService

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

SESSION_ID = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
MSG_ID_1 = uuid.UUID("660e8400-e29b-41d4-a716-446655440001")
MSG_ID_2 = uuid.UUID("660e8400-e29b-41d4-a716-446655440002")
NOW = datetime(2026, 3, 14, 12, 30, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _override_db() -> Any:
    """Dependency override yielding a mock session."""
    session = AsyncMock()
    session.execute = AsyncMock(return_value=None)
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise


def _make_message(
    msg_id: uuid.UUID,
    sequence: int,
    msg_type: str = "user_question",
) -> Any:
    """Build a mock Message ORM entity with loaded relations."""
    from unittest.mock import MagicMock

    from tabletop_oracle.models.enums import MessageType

    msg = MagicMock(spec=Message)
    msg.id = msg_id
    msg.session_id = SESSION_ID
    msg.type = MessageType(msg_type)
    msg.content = f"Message {sequence}"
    msg.sequence = sequence
    msg.in_reply_to_id = None
    msg.confidence_score = 0.9 if msg_type == "ai_answer" else None
    msg.processing_duration_ms = 500 if msg_type == "ai_answer" else None
    msg.created_at = NOW
    msg.citations = []
    msg.context_attachments = []
    return msg


# ---------------------------------------------------------------------------
# Tests: GET /sessions/{session_id}/messages
# ---------------------------------------------------------------------------


class TestListMessages:
    """Tests for GET /api/v1/sessions/{session_id}/messages."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> Any:
        """Override dependencies and enable bypass_auth for all tests."""
        from tabletop_oracle.api.deps import get_db

        app.dependency_overrides[get_db] = _override_db
        yield
        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_returns_200_with_messages(self) -> None:
        """Endpoint returns 200 with paginated message list."""
        msg1 = _make_message(MSG_ID_1, 1, "user_question")
        msg2 = _make_message(MSG_ID_2, 2, "ai_answer")

        with (
            patch("tabletop_oracle.auth.middleware.settings") as mock_settings,
            patch.object(MessageService, "list_messages", new_callable=AsyncMock) as mock_list,
        ):
            mock_settings.bypass_auth = True
            mock_list.return_value = ([msg1, msg2], 2)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    f"/api/v1/sessions/{SESSION_ID}/messages",
                    headers={"X-Request-ID": "req-test-1"},
                )

        assert response.status_code == 200
        body = response.json()
        assert "data" in body
        assert "meta" in body
        assert body["meta"]["request_id"] == "req-test-1"
        assert len(body["data"]) == 2

    @pytest.mark.asyncio
    async def test_pagination_metadata(self) -> None:
        """Response includes correct pagination metadata."""
        msg1 = _make_message(MSG_ID_1, 1)

        with (
            patch("tabletop_oracle.auth.middleware.settings") as mock_settings,
            patch.object(MessageService, "list_messages", new_callable=AsyncMock) as mock_list,
        ):
            mock_settings.bypass_auth = True
            mock_list.return_value = ([msg1], 30)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    f"/api/v1/sessions/{SESSION_ID}/messages",
                    params={"page": 1, "page_size": 10},
                )

        pagination = response.json()["meta"]["pagination"]
        assert pagination["page"] == 1
        assert pagination["page_size"] == 10
        assert pagination["total_count"] == 30
        assert pagination["total_pages"] == 3

    @pytest.mark.asyncio
    async def test_empty_session_returns_empty_list(self) -> None:
        """Endpoint returns empty list for session with no messages."""
        with (
            patch("tabletop_oracle.auth.middleware.settings") as mock_settings,
            patch.object(MessageService, "list_messages", new_callable=AsyncMock) as mock_list,
        ):
            mock_settings.bypass_auth = True
            mock_list.return_value = ([], 0)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    f"/api/v1/sessions/{SESSION_ID}/messages",
                )

        assert response.status_code == 200
        body = response.json()
        assert body["data"] == []
        assert body["meta"]["pagination"]["total_count"] == 0

    @pytest.mark.asyncio
    async def test_order_desc_accepted(self) -> None:
        """Endpoint accepts order=desc query parameter."""
        with (
            patch("tabletop_oracle.auth.middleware.settings") as mock_settings,
            patch.object(MessageService, "list_messages", new_callable=AsyncMock) as mock_list,
        ):
            mock_settings.bypass_auth = True
            mock_list.return_value = ([], 0)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    f"/api/v1/sessions/{SESSION_ID}/messages",
                    params={"order": "desc"},
                )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_invalid_order_rejected(self) -> None:
        """Endpoint rejects invalid order values."""
        with (
            patch("tabletop_oracle.auth.middleware.settings") as mock_settings,
        ):
            mock_settings.bypass_auth = True

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    f"/api/v1/sessions/{SESSION_ID}/messages",
                    params={"order": "invalid"},
                )

        # App returns 400 (custom error handler normalises validation errors)
        assert response.status_code in (400, 422)

    @pytest.mark.asyncio
    async def test_message_data_structure(self) -> None:
        """Message response data matches the expected contract."""
        msg = _make_message(MSG_ID_1, 1, "user_question")

        with (
            patch("tabletop_oracle.auth.middleware.settings") as mock_settings,
            patch.object(MessageService, "list_messages", new_callable=AsyncMock) as mock_list,
        ):
            mock_settings.bypass_auth = True
            mock_list.return_value = ([msg], 1)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    f"/api/v1/sessions/{SESSION_ID}/messages",
                )

        data = response.json()["data"][0]
        assert data["id"] == str(MSG_ID_1)
        assert data["session_id"] == str(SESSION_ID)
        assert data["type"] == "user_question"
        assert data["content"] == "Message 1"
        assert data["sequence"] == 1
        assert data["citations"] == []
        assert data["context_attachments"] == []

    @pytest.mark.asyncio
    async def test_ai_answer_includes_confidence(self) -> None:
        """AI answer messages include confidence_score and processing_duration_ms."""
        msg = _make_message(MSG_ID_1, 2, "ai_answer")

        with (
            patch("tabletop_oracle.auth.middleware.settings") as mock_settings,
            patch.object(MessageService, "list_messages", new_callable=AsyncMock) as mock_list,
        ):
            mock_settings.bypass_auth = True
            mock_list.return_value = ([msg], 1)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    f"/api/v1/sessions/{SESSION_ID}/messages",
                )

        data = response.json()["data"][0]
        assert data["confidence_score"] == 0.9
        assert data["processing_duration_ms"] == 500

    @pytest.mark.asyncio
    async def test_session_not_found_returns_404(self) -> None:
        """Endpoint returns 404 when session does not exist."""
        from tabletop_oracle.errors.exceptions import NotFoundError

        with (
            patch("tabletop_oracle.auth.middleware.settings") as mock_settings,
            patch.object(
                MessageService,
                "list_messages",
                new_callable=AsyncMock,
                side_effect=NotFoundError("Session", str(SESSION_ID)),
            ),
        ):
            mock_settings.bypass_auth = True

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    f"/api/v1/sessions/{SESSION_ID}/messages",
                )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_citations_included_for_ai_answer(self) -> None:
        """Citations are serialized in the response for ai_answer messages."""
        from unittest.mock import MagicMock

        from tabletop_oracle.models.enums import DocumentType, MessageType

        msg = MagicMock(spec=Message)
        msg.id = MSG_ID_1
        msg.session_id = SESSION_ID
        msg.type = MessageType.AI_ANSWER
        msg.content = "Answer"
        msg.sequence = 2
        msg.in_reply_to_id = None
        msg.confidence_score = 0.85
        msg.processing_duration_ms = 200
        msg.created_at = NOW
        msg.context_attachments = []

        citation = MagicMock()
        citation.id = uuid.uuid4()
        citation.document_id = uuid.uuid4()
        citation.document_name = "Core Rules"
        citation.document_type = DocumentType.CORE_RULES
        citation.section_path = "Chapter 1"
        citation.page_number = 10
        citation.excerpt = "Rule text here."
        msg.citations = [citation]

        with (
            patch("tabletop_oracle.auth.middleware.settings") as mock_settings,
            patch.object(MessageService, "list_messages", new_callable=AsyncMock) as mock_list,
        ):
            mock_settings.bypass_auth = True
            mock_list.return_value = ([msg], 1)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    f"/api/v1/sessions/{SESSION_ID}/messages",
                )

        data = response.json()["data"][0]
        assert len(data["citations"]) == 1
        assert data["citations"][0]["document_name"] == "Core Rules"
        assert data["citations"][0]["page_number"] == 10

    @pytest.mark.asyncio
    async def test_context_attachments_included_for_user_question(self) -> None:
        """Context attachments are serialized for user_question messages."""
        from unittest.mock import MagicMock

        from tabletop_oracle.models.enums import ContextAttachmentType, MessageType

        msg = MagicMock(spec=Message)
        msg.id = MSG_ID_1
        msg.session_id = SESSION_ID
        msg.type = MessageType.USER_QUESTION
        msg.content = "Question"
        msg.sequence = 1
        msg.in_reply_to_id = None
        msg.confidence_score = None
        msg.processing_duration_ms = None
        msg.created_at = NOW
        msg.citations = []

        attachment = MagicMock()
        attachment.id = uuid.uuid4()
        attachment.type = ContextAttachmentType.TEXT
        attachment.content = "Extra context"
        attachment.file_path = None
        attachment.file_name = None
        attachment.file_size = None
        attachment.processed_description = None
        msg.context_attachments = [attachment]

        with (
            patch("tabletop_oracle.auth.middleware.settings") as mock_settings,
            patch.object(MessageService, "list_messages", new_callable=AsyncMock) as mock_list,
        ):
            mock_settings.bypass_auth = True
            mock_list.return_value = ([msg], 1)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    f"/api/v1/sessions/{SESSION_ID}/messages",
                )

        data = response.json()["data"][0]
        assert len(data["context_attachments"]) == 1
        assert data["context_attachments"][0]["type"] == "text"
        assert data["context_attachments"][0]["content"] == "Extra context"
