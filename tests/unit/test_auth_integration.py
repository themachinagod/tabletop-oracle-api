"""Integration tests for session middleware — full request lifecycle.

Uses a minimal FastAPI test app with the session middleware to verify
end-to-end behavior: valid sessions, invalid sessions, expired sessions,
bypass mode, public paths, cookie clearing, and structlog user_id binding.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Depends, FastAPI
from httpx import ASGITransport, AsyncClient

from tabletop_oracle.auth.dependencies import CurrentUserDep, require_role
from tabletop_oracle.auth.middleware import SessionMiddleware
from tabletop_oracle.errors.handlers import register_exception_handlers
from tabletop_oracle.middleware.correlation import CorrelationMiddleware
from tabletop_oracle.models.enums import UserRole, UserStatus


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
    session_id: str = "valid-session-token",
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


def _build_app(*, bypass_auth: bool = False) -> FastAPI:
    """Build a minimal test app with session middleware and error handlers."""
    app = FastAPI()
    register_exception_handlers(app)
    app.add_middleware(SessionMiddleware)
    app.add_middleware(CorrelationMiddleware)

    @app.get("/api/v1/auth/me")
    async def auth_me() -> dict[str, str]:
        return {"status": "public-auth-endpoint"}

    @app.get("/api/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy"}

    @app.get("/api/v1/protected")
    async def protected(user: CurrentUserDep) -> dict[str, Any]:
        return {
            "user_id": str(user.user_id),
            "role": user.role,
            "email": user.email,
        }

    @app.get("/api/v1/curator-only", dependencies=[Depends(require_role("curator"))])
    async def curator_only() -> dict[str, str]:
        return {"area": "admin"}

    @app.get("/api/v1/player-ok", dependencies=[Depends(require_role("player"))])
    async def player_ok() -> dict[str, str]:
        return {"area": "player"}

    return app


#: A valid 43-char URL-safe base64 token for tests (matches _SESSION_ID_PATTERN).
_VALID_TOKEN = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnop0"


def _mock_auth_context(
    user_id: uuid.UUID,
    role: UserRole = UserRole.PLAYER,
    status: UserStatus = UserStatus.ACTIVE,
    expired: bool = False,
):
    """Context manager that patches async_session_factory, SessionStore.get/touch."""
    mock_session = _make_mock_session(session_id=_VALID_TOKEN, user_id=user_id, expired=expired)
    mock_user = _make_mock_user(user_id=user_id, role=role, status=status)

    mock_db = AsyncMock()
    mock_db.get = AsyncMock(return_value=mock_user)
    mock_db.commit = AsyncMock()

    mock_factory = MagicMock()
    mock_factory.__aenter__ = AsyncMock(return_value=mock_db)
    mock_factory.__aexit__ = AsyncMock(return_value=False)

    class _Ctx:
        def __init__(self):
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
            self.mock_db = mock_db
            self.mock_touch: AsyncMock | None = None

        def __enter__(self):
            self._factory_patch.__enter__()
            self._get_patch.__enter__()
            self.mock_touch = self._touch_patch.__enter__()
            return self

        def __exit__(self, *args):
            self._touch_patch.__exit__(*args)
            self._get_patch.__exit__(*args)
            self._factory_patch.__exit__(*args)

    return _Ctx()


@pytest.fixture(autouse=True)
def _disable_bypass_auth():
    """Ensure bypass_auth is disabled for all tests unless explicitly overridden."""
    with patch("tabletop_oracle.auth.middleware.settings") as mock_settings:
        mock_settings.bypass_auth = False
        mock_settings.session_cookie_secure = False
        yield mock_settings


@pytest.fixture
def test_app() -> FastAPI:
    """Provide a test FastAPI app with session middleware."""
    return _build_app()


class TestPublicEndpoints:
    """Public endpoints should not require authentication."""

    @pytest.mark.asyncio
    async def test_health_no_auth_required(self, test_app: FastAPI) -> None:
        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/v1/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_auth_endpoints_no_auth_required(self, test_app: FastAPI) -> None:
        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/v1/auth/me")
        assert resp.status_code == 200


class TestMissingSession:
    """Protected endpoints without a session should return 401."""

    @pytest.mark.asyncio
    async def test_no_cookie_no_header_returns_401(self, test_app: FastAPI) -> None:
        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/v1/protected")
        assert resp.status_code == 401
        body = resp.json()
        assert body["error"]["code"] == "AUTHENTICATION_REQUIRED"

    @pytest.mark.asyncio
    async def test_401_includes_request_id(self, test_app: FastAPI) -> None:
        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/v1/protected",
                headers={"X-Request-ID": "test-corr-id"},
            )
        body = resp.json()
        assert body["meta"]["request_id"] == "test-corr-id"


class TestInvalidSession:
    """Sessions that don't exist in the store should return 401."""

    @pytest.mark.asyncio
    async def test_unknown_session_returns_401(self, test_app: FastAPI) -> None:
        mock_db = AsyncMock()
        mock_db.get = AsyncMock(return_value=None)

        mock_factory = MagicMock()
        mock_factory.__aenter__ = AsyncMock(return_value=mock_db)
        mock_factory.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "tabletop_oracle.auth.middleware.async_session_factory",
                return_value=mock_factory,
            ),
            patch(
                "tabletop_oracle.auth.middleware.SessionStore.get",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            transport = ASGITransport(app=test_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/protected",
                    cookies={"sid": _VALID_TOKEN},
                )
        assert resp.status_code == 401


class TestExpiredSession:
    """Expired sessions should return 401 and clear cookie."""

    @pytest.mark.asyncio
    async def test_expired_session_returns_401_and_clears_cookie(self, test_app: FastAPI) -> None:
        user_id = uuid.uuid4()
        expired_session = _make_mock_session(
            session_id=_VALID_TOKEN, user_id=user_id, expired=True
        )

        mock_db = AsyncMock()
        mock_db.commit = AsyncMock()

        mock_factory = MagicMock()
        mock_factory.__aenter__ = AsyncMock(return_value=mock_db)
        mock_factory.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "tabletop_oracle.auth.middleware.async_session_factory",
                return_value=mock_factory,
            ),
            patch(
                "tabletop_oracle.auth.middleware.SessionStore.get",
                new_callable=AsyncMock,
                return_value=expired_session,
            ),
        ):
            transport = ASGITransport(app=test_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/protected",
                    cookies={"sid": _VALID_TOKEN},
                )
        assert resp.status_code == 401
        set_cookie = resp.headers.get("set-cookie", "")
        assert "sid" in set_cookie.lower()


class TestValidSession:
    """Valid session should inject CurrentUser and return success."""

    @pytest.mark.asyncio
    async def test_valid_session_via_cookie(self, test_app: FastAPI) -> None:
        user_id = uuid.uuid4()
        mock_session = _make_mock_session(session_id=_VALID_TOKEN, user_id=user_id)
        mock_user = _make_mock_user(user_id=user_id, role=UserRole.PLAYER)

        mock_db = AsyncMock()
        mock_db.get = AsyncMock(return_value=mock_user)
        mock_db.commit = AsyncMock()

        mock_factory = MagicMock()
        mock_factory.__aenter__ = AsyncMock(return_value=mock_db)
        mock_factory.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "tabletop_oracle.auth.middleware.async_session_factory",
                return_value=mock_factory,
            ),
            patch(
                "tabletop_oracle.auth.middleware.SessionStore.get",
                new_callable=AsyncMock,
                return_value=mock_session,
            ),
            patch(
                "tabletop_oracle.auth.middleware.SessionStore.touch",
                new_callable=AsyncMock,
            ) as mock_touch,
        ):
            transport = ASGITransport(app=test_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/protected",
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 200
        body = resp.json()
        assert body["user_id"] == str(user_id)
        assert body["role"] == "player"
        mock_touch.assert_called_once_with(mock_db, _VALID_TOKEN)

    @pytest.mark.asyncio
    async def test_valid_session_via_bearer(self, test_app: FastAPI) -> None:
        user_id = uuid.uuid4()
        mock_session = _make_mock_session(session_id=_VALID_TOKEN, user_id=user_id)
        mock_user = _make_mock_user(user_id=user_id, role=UserRole.CURATOR)

        mock_db = AsyncMock()
        mock_db.get = AsyncMock(return_value=mock_user)
        mock_db.commit = AsyncMock()

        mock_factory = MagicMock()
        mock_factory.__aenter__ = AsyncMock(return_value=mock_db)
        mock_factory.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "tabletop_oracle.auth.middleware.async_session_factory",
                return_value=mock_factory,
            ),
            patch(
                "tabletop_oracle.auth.middleware.SessionStore.get",
                new_callable=AsyncMock,
                return_value=mock_session,
            ),
            patch(
                "tabletop_oracle.auth.middleware.SessionStore.touch",
                new_callable=AsyncMock,
            ),
        ):
            transport = ASGITransport(app=test_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/protected",
                    headers={"Authorization": f"Bearer {_VALID_TOKEN}"},
                )

        assert resp.status_code == 200
        body = resp.json()
        assert body["role"] == "curator"

    @pytest.mark.asyncio
    async def test_malformed_session_id_returns_401(self, test_app: FastAPI) -> None:
        """Session IDs that don't match the expected format are rejected."""
        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/v1/protected",
                cookies={"sid": "too-short"},
            )
        assert resp.status_code == 401


class TestDisabledUser:
    """Disabled users should be rejected even with a valid session."""

    @pytest.mark.asyncio
    async def test_disabled_user_returns_401(self, test_app: FastAPI) -> None:
        user_id = uuid.uuid4()
        with _mock_auth_context(user_id, status=UserStatus.DISABLED):
            transport = ASGITransport(app=test_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/protected",
                    cookies={"sid": _VALID_TOKEN},
                )
        assert resp.status_code == 401


class TestBypassAuth:
    """bypass_auth setting should inject a mock user without DB access."""

    @pytest.mark.asyncio
    async def test_bypass_injects_mock_curator(self, _disable_bypass_auth: MagicMock) -> None:
        _disable_bypass_auth.bypass_auth = True
        app = _build_app(bypass_auth=True)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/v1/protected")
        assert resp.status_code == 200
        body = resp.json()
        assert body["role"] == "curator"
        assert body["email"] == "dev@localhost"


class TestRoleEnforcementIntegration:
    """Role enforcement via require_role dependency in route context."""

    @pytest.mark.asyncio
    async def test_curator_accesses_curator_route(self, test_app: FastAPI) -> None:
        user_id = uuid.uuid4()
        with _mock_auth_context(user_id, role=UserRole.CURATOR):
            transport = ASGITransport(app=test_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/curator-only",
                    cookies={"sid": _VALID_TOKEN},
                )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_player_rejected_from_curator_route(self, test_app: FastAPI) -> None:
        user_id = uuid.uuid4()
        with _mock_auth_context(user_id, role=UserRole.PLAYER):
            transport = ASGITransport(app=test_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/curator-only",
                    cookies={"sid": _VALID_TOKEN},
                )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_curator_accesses_player_route(self, test_app: FastAPI) -> None:
        user_id = uuid.uuid4()
        with _mock_auth_context(user_id, role=UserRole.CURATOR):
            transport = ASGITransport(app=test_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/player-ok",
                    cookies={"sid": _VALID_TOKEN},
                )
        assert resp.status_code == 200


class TestStructlogBinding:
    """Structlog user_id binding during authenticated requests."""

    @pytest.mark.asyncio
    async def test_user_id_bound_to_structlog(self, test_app: FastAPI) -> None:
        user_id = uuid.uuid4()
        bound_vars: dict[str, Any] = {}

        def capture_bind(**kwargs: Any) -> None:
            bound_vars.update(kwargs)

        mock_session = _make_mock_session(session_id=_VALID_TOKEN, user_id=user_id)
        mock_user = _make_mock_user(user_id=user_id)

        mock_db = AsyncMock()
        mock_db.get = AsyncMock(return_value=mock_user)
        mock_db.commit = AsyncMock()

        mock_factory = MagicMock()
        mock_factory.__aenter__ = AsyncMock(return_value=mock_db)
        mock_factory.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "tabletop_oracle.auth.middleware.async_session_factory",
                return_value=mock_factory,
            ),
            patch(
                "tabletop_oracle.auth.middleware.SessionStore.get",
                new_callable=AsyncMock,
                return_value=mock_session,
            ),
            patch(
                "tabletop_oracle.auth.middleware.SessionStore.touch",
                new_callable=AsyncMock,
            ),
            patch(
                "tabletop_oracle.auth.middleware.structlog.contextvars.bind_contextvars",
                side_effect=capture_bind,
            ),
        ):
            transport = ASGITransport(app=test_app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                await client.get(
                    "/api/v1/protected",
                    cookies={"sid": _VALID_TOKEN},
                )

        assert bound_vars.get("user_id") == str(user_id)
