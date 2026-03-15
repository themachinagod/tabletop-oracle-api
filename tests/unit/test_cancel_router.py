"""Unit tests for the message cancellation API endpoint.

Tests DELETE /api/v1/sessions/{session_id}/messages/{message_id}
through the ASGI transport with bypass_auth enabled. Verifies
202/204 responses, F001 envelope structure, and error handling.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from tabletop_oracle.main import app
from tabletop_oracle.services.message_service import MessageService

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

SESSION_ID = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
MESSAGE_ID = uuid.UUID("660e8400-e29b-41d4-a716-446655440001")


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


# ---------------------------------------------------------------------------
# Tests: DELETE /sessions/{session_id}/messages/{message_id}
# ---------------------------------------------------------------------------


class TestCancelMessage:
    """Tests for DELETE /api/v1/sessions/{session_id}/messages/{message_id}."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> Any:
        """Override dependencies and enable bypass_auth for all tests."""
        from tabletop_oracle.api.deps import get_db

        app.dependency_overrides[get_db] = _override_db
        yield
        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_returns_202_for_active_query(self) -> None:
        """Endpoint returns 202 Accepted with cancel response for active query."""
        message = MagicMock()
        message.id = MESSAGE_ID

        with (
            patch("tabletop_oracle.auth.middleware.settings") as mock_settings,
            patch.object(
                MessageService,
                "cancel_message",
                new_callable=AsyncMock,
            ) as mock_cancel,
        ):
            mock_settings.bypass_auth = True
            mock_cancel.return_value = (message, True)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.delete(
                    f"/api/v1/sessions/{SESSION_ID}/messages/{MESSAGE_ID}",
                    headers={"X-Request-ID": "req-cancel-1"},
                )

        assert response.status_code == 202
        body = response.json()
        assert "data" in body
        assert body["data"]["id"] == str(MESSAGE_ID)
        assert body["data"]["status"] == "cancelling"
        assert body["meta"]["request_id"] == "req-cancel-1"

    @pytest.mark.asyncio
    async def test_returns_204_for_complete_message(self) -> None:
        """Endpoint returns 204 No Content for already-complete message."""
        message = MagicMock()
        message.id = MESSAGE_ID

        with (
            patch("tabletop_oracle.auth.middleware.settings") as mock_settings,
            patch.object(
                MessageService,
                "cancel_message",
                new_callable=AsyncMock,
            ) as mock_cancel,
        ):
            mock_settings.bypass_auth = True
            mock_cancel.return_value = (message, False)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.delete(
                    f"/api/v1/sessions/{SESSION_ID}/messages/{MESSAGE_ID}",
                )

        assert response.status_code == 204

    @pytest.mark.asyncio
    async def test_returns_404_for_missing_session(self) -> None:
        """Endpoint returns 404 when session does not exist."""
        from tabletop_oracle.errors.exceptions import NotFoundError

        with (
            patch("tabletop_oracle.auth.middleware.settings") as mock_settings,
            patch.object(
                MessageService,
                "cancel_message",
                new_callable=AsyncMock,
                side_effect=NotFoundError("Session", str(SESSION_ID)),
            ),
        ):
            mock_settings.bypass_auth = True

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.delete(
                    f"/api/v1/sessions/{SESSION_ID}/messages/{MESSAGE_ID}",
                )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_404_for_missing_message(self) -> None:
        """Endpoint returns 404 when message does not exist."""
        from tabletop_oracle.errors.exceptions import NotFoundError

        with (
            patch("tabletop_oracle.auth.middleware.settings") as mock_settings,
            patch.object(
                MessageService,
                "cancel_message",
                new_callable=AsyncMock,
                side_effect=NotFoundError("Message", str(MESSAGE_ID)),
            ),
        ):
            mock_settings.bypass_auth = True

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.delete(
                    f"/api/v1/sessions/{SESSION_ID}/messages/{MESSAGE_ID}",
                )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_cancel_response_envelope_structure(self) -> None:
        """202 response follows the F001 DataEnvelope convention."""
        message = MagicMock()
        message.id = MESSAGE_ID

        with (
            patch("tabletop_oracle.auth.middleware.settings") as mock_settings,
            patch.object(
                MessageService,
                "cancel_message",
                new_callable=AsyncMock,
            ) as mock_cancel,
        ):
            mock_settings.bypass_auth = True
            mock_cancel.return_value = (message, True)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.delete(
                    f"/api/v1/sessions/{SESSION_ID}/messages/{MESSAGE_ID}",
                    headers={"X-Request-ID": "req-env-check"},
                )

        body = response.json()
        # F001 envelope: { data, meta: { request_id } }
        assert set(body.keys()) == {"data", "meta"}
        assert "request_id" in body["meta"]
        assert set(body["data"].keys()) == {"id", "status"}
