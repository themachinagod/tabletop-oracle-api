"""Integration tests for SSE endpoint authentication and authorisation.

Tests verify that SSE endpoints properly integrate with SessionMiddleware
for cookie-based auth, enforce resource-level authorisation (session
ownership for AI streams, curator role for document streams), and return
standard JSON error responses for pre-stream failures.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from tabletop_oracle.api.sse import (
    _active_document_streams,
    _active_message_streams,
    _is_session_valid,
    _session_expiry_wrapper,
    router,
)
from tabletop_oracle.auth.middleware import SessionMiddleware
from tabletop_oracle.errors.handlers import register_exception_handlers
from tabletop_oracle.middleware.correlation import CorrelationMiddleware
from tabletop_oracle.models.enums import UserRole, UserStatus
from tabletop_oracle.streaming.events import SseEvent

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: A valid 43-char URL-safe base64 token matching _SESSION_ID_PATTERN.
_VALID_TOKEN = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnop0"

_SESSION_UUID = uuid.uuid4()
_MESSAGE_UUID = uuid.uuid4()
_DOCUMENT_UUID = uuid.uuid4()


def _make_mock_user(
    user_id: uuid.UUID | None = None,
    role: UserRole = UserRole.PLAYER,
    status: UserStatus = UserStatus.ACTIVE,
    email: str = "player@example.com",
    display_name: str = "Player",
) -> MagicMock:
    """Create a mock User ORM object."""
    user = MagicMock()
    user.id = user_id or uuid.uuid4()
    user.role = role
    user.status = status
    user.email = email
    user.display_name = display_name
    return user


def _make_mock_session(
    user_id: uuid.UUID | None = None,
    expired: bool = False,
) -> MagicMock:
    """Create a mock AuthSession object."""
    session = MagicMock()
    session.user_id = user_id or uuid.uuid4()
    if expired:
        session.expires_at = datetime.now(UTC) - timedelta(hours=1)
    else:
        session.expires_at = datetime.now(UTC) + timedelta(days=30)
    return session


def _mock_auth_context(
    user_id: uuid.UUID,
    role: UserRole = UserRole.PLAYER,
    status: UserStatus = UserStatus.ACTIVE,
    expired: bool = False,
) -> _MockAuthCtx:
    """Create a context manager that patches session middleware dependencies."""
    return _MockAuthCtx(user_id, role, status, expired)


class _MockAuthCtx:
    """Context manager patching async_session_factory and SessionStore for tests."""

    def __init__(
        self,
        user_id: uuid.UUID,
        role: UserRole,
        status: UserStatus,
        expired: bool,
    ) -> None:
        self._user_id = user_id
        self._role = role
        self._status = status
        self._expired = expired

    def __enter__(self) -> _MockAuthCtx:
        mock_session = _make_mock_session(user_id=self._user_id, expired=self._expired)
        mock_user = _make_mock_user(user_id=self._user_id, role=self._role, status=self._status)

        mock_db = AsyncMock()
        mock_db.get = AsyncMock(return_value=mock_user)
        mock_db.commit = AsyncMock()

        mock_factory = MagicMock()
        mock_factory.__aenter__ = AsyncMock(return_value=mock_db)
        mock_factory.__aexit__ = AsyncMock(return_value=False)

        self._factory_patch = patch(
            "tabletop_oracle.auth.middleware.async_session_factory",
            return_value=mock_factory,
        )
        self._get_patch = patch(
            "tabletop_oracle.auth.middleware.SessionStore.get",
            new_callable=AsyncMock,
            return_value=mock_session,
        )
        self._touch_patch = patch(
            "tabletop_oracle.auth.middleware.SessionStore.touch",
            new_callable=AsyncMock,
        )

        self._factory_patch.__enter__()
        self._get_patch.__enter__()
        self._touch_patch.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self._touch_patch.__exit__(*args)
        self._get_patch.__exit__(*args)
        self._factory_patch.__exit__(*args)


def _build_test_app() -> FastAPI:
    """Build a minimal FastAPI app with SSE routes and auth middleware."""
    app = FastAPI()
    register_exception_handlers(app)
    app.add_middleware(SessionMiddleware)
    app.add_middleware(CorrelationMiddleware)
    app.include_router(router, prefix="/api/v1")
    return app


@pytest.fixture(autouse=True)
def _disable_bypass_auth():
    """Ensure bypass_auth is disabled for all tests."""
    with patch("tabletop_oracle.auth.middleware.settings") as mock_settings:
        mock_settings.bypass_auth = False
        mock_settings.session_cookie_secure = False
        yield mock_settings


@pytest.fixture
def test_app() -> FastAPI:
    """Provide a test FastAPI app with SSE routes."""
    return _build_test_app()


@pytest.fixture(autouse=True)
def _clear_stream_registries():
    """Clear the stream registries before each test."""
    _active_message_streams.clear()
    _active_document_streams.clear()
    yield
    _active_message_streams.clear()
    _active_document_streams.clear()


# ---------------------------------------------------------------------------
# AI Response Stream — Authentication
# ---------------------------------------------------------------------------


class TestMessageStreamAuthentication:
    """AI response stream endpoint requires valid session cookie."""

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, test_app: FastAPI) -> None:
        """Unauthenticated request returns 401 AUTHENTICATION_REQUIRED as JSON."""
        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                f"/api/v1/sessions/{_SESSION_UUID}/messages/{_MESSAGE_UUID}/stream",
            )
        assert resp.status_code == 401
        body = resp.json()
        assert body["error"]["code"] == "AUTHENTICATION_REQUIRED"
        assert resp.headers["content-type"] == "application/json"

    @pytest.mark.asyncio
    async def test_expired_session_returns_401(self, test_app: FastAPI) -> None:
        """Expired session cookie returns 401."""
        user_id = uuid.uuid4()
        with _mock_auth_context(user_id, expired=True):
            transport = ASGITransport(app=test_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    f"/api/v1/sessions/{_SESSION_UUID}/messages/{_MESSAGE_UUID}/stream",
                    cookies={"sid": _VALID_TOKEN},
                )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_valid_session_no_stream_returns_204(self, test_app: FastAPI) -> None:
        """Authenticated request for non-streaming message returns 204 No Content."""
        user_id = uuid.uuid4()
        with _mock_auth_context(user_id, role=UserRole.PLAYER):
            transport = ASGITransport(app=test_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    f"/api/v1/sessions/{_SESSION_UUID}/messages/{_MESSAGE_UUID}/stream",
                    cookies={"sid": _VALID_TOKEN},
                )
        assert resp.status_code == 204

    @pytest.mark.asyncio
    async def test_bearer_auth_accepted(self, test_app: FastAPI) -> None:
        """Bearer token auth is accepted for SSE endpoints."""
        user_id = uuid.uuid4()
        with _mock_auth_context(user_id, role=UserRole.PLAYER):
            transport = ASGITransport(app=test_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    f"/api/v1/sessions/{_SESSION_UUID}/messages/{_MESSAGE_UUID}/stream",
                    headers={"Authorization": f"Bearer {_VALID_TOKEN}"},
                )
        assert resp.status_code == 204

    @pytest.mark.asyncio
    async def test_malformed_session_id_returns_401(self, test_app: FastAPI) -> None:
        """Malformed session cookie returns 401."""
        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                f"/api/v1/sessions/{_SESSION_UUID}/messages/{_MESSAGE_UUID}/stream",
                cookies={"sid": "too-short"},
            )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# AI Response Stream — 204 No Content
# ---------------------------------------------------------------------------


class TestMessageStreamNoContent:
    """Non-streaming messages return 204 No Content."""

    @pytest.mark.asyncio
    async def test_no_active_stream_returns_204(self, test_app: FastAPI) -> None:
        """When no stream is active for the message, return 204."""
        user_id = uuid.uuid4()
        with _mock_auth_context(user_id, role=UserRole.PLAYER):
            transport = ASGITransport(app=test_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    f"/api/v1/sessions/{_SESSION_UUID}/messages/{_MESSAGE_UUID}/stream",
                    cookies={"sid": _VALID_TOKEN},
                )
        assert resp.status_code == 204


# ---------------------------------------------------------------------------
# AI Response Stream — Active Stream
# ---------------------------------------------------------------------------


class TestMessageStreamActive:
    """When a stream is active, return SSE event-stream response."""

    @pytest.mark.asyncio
    async def test_active_stream_returns_event_stream(self, test_app: FastAPI) -> None:
        """Active stream returns 200 with text/event-stream content type."""
        user_id = uuid.uuid4()
        # Register an active stream for this session+message pair.
        _active_message_streams[(_SESSION_UUID, _MESSAGE_UUID)] = type(None)

        with (
            _mock_auth_context(user_id, role=UserRole.PLAYER),
            patch(
                "tabletop_oracle.api.sse._is_session_valid",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            transport = ASGITransport(app=test_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    f"/api/v1/sessions/{_SESSION_UUID}/messages/{_MESSAGE_UUID}/stream",
                    cookies={"sid": _VALID_TOKEN},
                )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# Document Processing Stream — Authentication & Authorisation
# ---------------------------------------------------------------------------


class TestDocumentStreamAuthentication:
    """Document processing stream endpoint requires curator role."""

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, test_app: FastAPI) -> None:
        """Unauthenticated request returns 401 as JSON."""
        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                f"/api/v1/documents/{_DOCUMENT_UUID}/processing/stream",
            )
        assert resp.status_code == 401
        body = resp.json()
        assert body["error"]["code"] == "AUTHENTICATION_REQUIRED"
        assert resp.headers["content-type"] == "application/json"

    @pytest.mark.asyncio
    async def test_player_returns_403(self, test_app: FastAPI) -> None:
        """Non-curator (player) returns 403 FORBIDDEN as JSON."""
        user_id = uuid.uuid4()
        with _mock_auth_context(user_id, role=UserRole.PLAYER):
            transport = ASGITransport(app=test_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    f"/api/v1/documents/{_DOCUMENT_UUID}/processing/stream",
                    cookies={"sid": _VALID_TOKEN},
                )
        assert resp.status_code == 403
        body = resp.json()
        assert body["error"]["code"] == "FORBIDDEN"

    @pytest.mark.asyncio
    async def test_curator_no_stream_returns_204(self, test_app: FastAPI) -> None:
        """Curator accessing non-processing document returns 204."""
        user_id = uuid.uuid4()
        with _mock_auth_context(user_id, role=UserRole.CURATOR):
            transport = ASGITransport(app=test_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    f"/api/v1/documents/{_DOCUMENT_UUID}/processing/stream",
                    cookies={"sid": _VALID_TOKEN},
                )
        assert resp.status_code == 204

    @pytest.mark.asyncio
    async def test_curator_active_stream_returns_event_stream(self, test_app: FastAPI) -> None:
        """Curator accessing actively processing document returns SSE stream."""
        user_id = uuid.uuid4()
        _active_document_streams[_DOCUMENT_UUID] = type(None)

        with (
            _mock_auth_context(user_id, role=UserRole.CURATOR),
            patch(
                "tabletop_oracle.api.sse._is_session_valid",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            transport = ASGITransport(app=test_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    f"/api/v1/documents/{_DOCUMENT_UUID}/processing/stream",
                    cookies={"sid": _VALID_TOKEN},
                )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    @pytest.mark.asyncio
    async def test_expired_session_returns_401(self, test_app: FastAPI) -> None:
        """Expired session returns 401 for document stream."""
        user_id = uuid.uuid4()
        with _mock_auth_context(user_id, role=UserRole.CURATOR, expired=True):
            transport = ASGITransport(app=test_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    f"/api/v1/documents/{_DOCUMENT_UUID}/processing/stream",
                    cookies={"sid": _VALID_TOKEN},
                )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Session Expiry During Active Stream
# ---------------------------------------------------------------------------


class TestSessionExpiryDuringStream:
    """Session expiry during an active stream should close the connection."""

    @pytest.mark.asyncio
    async def test_session_expiry_wrapper_closes_on_invalid_session(self) -> None:
        """_session_expiry_wrapper terminates when session becomes invalid."""

        async def _slow_generator() -> AsyncGenerator[SseEvent, None]:
            """Yield events with small delays to allow expiry check."""
            for i in range(10):
                yield SseEvent(type="stream.token", payload={"token": f"t{i}"})
                await asyncio.sleep(0.01)

        events: list[SseEvent] = []
        # Session becomes invalid after the first check.
        with patch(
            "tabletop_oracle.api.sse._is_session_valid",
            new_callable=AsyncMock,
            return_value=False,
        ):
            async for event in _session_expiry_wrapper(
                _slow_generator(),
                "test-session-id",
                check_interval=0.0,  # Check on every event.
            ):
                events.append(event)

        # Should have received the first event but stopped after the check.
        assert len(events) >= 1
        assert len(events) < 10

    @pytest.mark.asyncio
    async def test_session_expiry_wrapper_passes_through_on_valid_session(self) -> None:
        """_session_expiry_wrapper passes all events when session stays valid."""

        async def _generator() -> AsyncGenerator[SseEvent, None]:
            for i in range(5):
                yield SseEvent(type="stream.token", payload={"token": f"t{i}"})

        events: list[SseEvent] = []
        with patch(
            "tabletop_oracle.api.sse._is_session_valid",
            new_callable=AsyncMock,
            return_value=True,
        ):
            async for event in _session_expiry_wrapper(
                _generator(),
                "test-session-id",
                check_interval=0.0,
            ):
                events.append(event)

        assert len(events) == 5


# ---------------------------------------------------------------------------
# Session Validity Check
# ---------------------------------------------------------------------------


class TestIsSessionValid:
    """Unit tests for the _is_session_valid helper."""

    @pytest.mark.asyncio
    async def test_valid_session_returns_true(self) -> None:
        """A session that exists and hasn't expired returns True."""
        mock_session = _make_mock_session()

        mock_db = AsyncMock()
        mock_factory = MagicMock()
        mock_factory.__aenter__ = AsyncMock(return_value=mock_db)
        mock_factory.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "tabletop_oracle.api.sse.async_session_factory",
                return_value=mock_factory,
            ),
            patch(
                "tabletop_oracle.api.sse.SessionStore.get",
                new_callable=AsyncMock,
                return_value=mock_session,
            ),
        ):
            result = await _is_session_valid("test-session-id")

        assert result is True

    @pytest.mark.asyncio
    async def test_missing_session_returns_false(self) -> None:
        """A missing session returns False."""
        mock_db = AsyncMock()
        mock_factory = MagicMock()
        mock_factory.__aenter__ = AsyncMock(return_value=mock_db)
        mock_factory.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "tabletop_oracle.api.sse.async_session_factory",
                return_value=mock_factory,
            ),
            patch(
                "tabletop_oracle.api.sse.SessionStore.get",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await _is_session_valid("test-session-id")

        assert result is False

    @pytest.mark.asyncio
    async def test_expired_session_returns_false(self) -> None:
        """An expired session returns False."""
        mock_session = _make_mock_session(expired=True)

        mock_db = AsyncMock()
        mock_factory = MagicMock()
        mock_factory.__aenter__ = AsyncMock(return_value=mock_db)
        mock_factory.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "tabletop_oracle.api.sse.async_session_factory",
                return_value=mock_factory,
            ),
            patch(
                "tabletop_oracle.api.sse.SessionStore.get",
                new_callable=AsyncMock,
                return_value=mock_session,
            ),
        ):
            result = await _is_session_valid("test-session-id")

        assert result is False


# ---------------------------------------------------------------------------
# Error Response Format
# ---------------------------------------------------------------------------


class TestErrorResponseFormat:
    """Verify error responses are JSON (not event-stream) with standard format."""

    @pytest.mark.asyncio
    async def test_401_response_is_json_not_event_stream(self, test_app: FastAPI) -> None:
        """401 from SessionMiddleware is application/json, not text/event-stream."""
        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                f"/api/v1/sessions/{_SESSION_UUID}/messages/{_MESSAGE_UUID}/stream",
            )
        assert resp.status_code == 401
        assert resp.headers["content-type"] == "application/json"
        body = resp.json()
        assert "error" in body
        assert "meta" in body

    @pytest.mark.asyncio
    async def test_403_response_is_json_not_event_stream(self, test_app: FastAPI) -> None:
        """403 for non-curator on document stream is JSON."""
        user_id = uuid.uuid4()
        with _mock_auth_context(user_id, role=UserRole.PLAYER):
            transport = ASGITransport(app=test_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    f"/api/v1/documents/{_DOCUMENT_UUID}/processing/stream",
                    cookies={"sid": _VALID_TOKEN},
                )
        assert resp.status_code == 403
        assert resp.headers["content-type"] == "application/json"
        body = resp.json()
        assert body["error"]["code"] == "FORBIDDEN"

    @pytest.mark.asyncio
    async def test_401_includes_request_id(self, test_app: FastAPI) -> None:
        """401 responses include the X-Request-ID for correlation."""
        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                f"/api/v1/sessions/{_SESSION_UUID}/messages/{_MESSAGE_UUID}/stream",
                headers={"X-Request-ID": "sse-test-corr-id"},
            )
        body = resp.json()
        assert body["meta"]["request_id"] == "sse-test-corr-id"
