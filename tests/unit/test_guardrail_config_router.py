"""Unit tests for the guardrail config admin API router.

Tests the ``/api/v1/admin/guardrail-config`` endpoints: reading config,
updating config, role enforcement, and missing config handling.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from tabletop_oracle.api.admin.guardrail_config import router as guardrail_config_router
from tabletop_oracle.api.deps import get_db
from tabletop_oracle.auth.middleware import SessionMiddleware
from tabletop_oracle.errors.exceptions import NotFoundError
from tabletop_oracle.errors.handlers import register_exception_handlers
from tabletop_oracle.middleware.correlation import CorrelationMiddleware
from tabletop_oracle.models.enums import UserRole, UserStatus
from tabletop_oracle.models.guardrail import GuardrailConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: Valid 43-char token matching the middleware's session ID pattern.
_VALID_TOKEN = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnop0"


def _make_config_entity(
    *,
    enforcement_enabled: bool = True,
    max_tokens_per_query: int | None = 50000,
    max_model_calls_per_query: int | None = 10,
    max_queries_per_session: int | None = 100,
    daily_token_budget: int | None = 1000000,
    daily_query_budget: int | None = 5000,
    per_document_ingestion_limit: int | None = 200000,
    updated_at: datetime | None = None,
) -> MagicMock:
    """Build a mock GuardrailConfig entity for test responses."""
    config = MagicMock(spec=GuardrailConfig)
    config.enforcement_enabled = enforcement_enabled
    config.max_tokens_per_query = max_tokens_per_query
    config.max_model_calls_per_query = max_model_calls_per_query
    config.max_queries_per_session = max_queries_per_session
    config.daily_token_budget = daily_token_budget
    config.daily_query_budget = daily_query_budget
    config.per_document_ingestion_limit = per_document_ingestion_limit
    config.updated_at = updated_at or datetime(2026, 3, 14, 10, 0, 0, tzinfo=UTC)
    return config


def _make_mock_session(
    user_id: uuid.UUID,
    expired: bool = False,
) -> MagicMock:
    """Create a mock AuthSession object."""
    from datetime import timedelta

    session = MagicMock()
    session.user_id = user_id
    if expired:
        session.expires_at = datetime.now(UTC) - timedelta(hours=1)
    else:
        session.expires_at = datetime.now(UTC) + timedelta(days=30)
    return session


def _mock_auth_context(role: UserRole = UserRole.CURATOR):
    """Context manager patching auth middleware for a user with given role.

    Disables ``bypass_auth`` so the middleware runs its full validation
    path, even in CI environments where bypass_auth may default to True.
    """
    user_id = uuid.UUID("12345678-1234-1234-1234-123456789abc")
    mock_session = _make_mock_session(user_id=user_id)
    mock_user = MagicMock()
    mock_user.id = user_id
    mock_user.role = role
    mock_user.status = UserStatus.ACTIVE
    mock_user.email = "test@example.com"
    mock_user.display_name = "Test User"

    mock_db = AsyncMock()
    mock_db.get = AsyncMock(return_value=mock_user)
    mock_db.commit = AsyncMock()

    mock_factory = MagicMock()
    mock_factory.__aenter__ = AsyncMock(return_value=mock_db)
    mock_factory.__aexit__ = AsyncMock(return_value=False)

    class _Ctx:
        def __init__(self) -> None:
            self._settings_patch = patch(
                "tabletop_oracle.auth.middleware.settings",
            )
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

        def __enter__(self) -> _Ctx:
            mock_settings = self._settings_patch.start()
            mock_settings.bypass_auth = False
            mock_settings.session_cookie_secure = False
            self._factory_patch.start()
            self._get_patch.start()
            self._touch_patch.start()
            return self

        def __exit__(self, *args: object) -> None:
            self._touch_patch.stop()
            self._get_patch.stop()
            self._factory_patch.stop()
            self._settings_patch.stop()

    return _Ctx()


def _build_test_app() -> FastAPI:
    """Build a minimal FastAPI app with admin guardrail-config router."""
    test_app = FastAPI()
    register_exception_handlers(test_app)
    test_app.add_middleware(SessionMiddleware)
    test_app.add_middleware(CorrelationMiddleware)
    test_app.include_router(
        guardrail_config_router,
        prefix="/api/v1/admin/guardrail-config",
    )
    return test_app


async def _mock_db_session():
    """Dependency override yielding a mock session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    yield session


# ---------------------------------------------------------------------------
# GET /api/v1/admin/guardrail-config
# ---------------------------------------------------------------------------


class TestGetGuardrailConfig:
    """Tests for GET /api/v1/admin/guardrail-config."""

    @pytest.mark.asyncio
    async def test_get_returns_config(self) -> None:
        """Curator can read the guardrail configuration."""
        config = _make_config_entity()

        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with (
            _mock_auth_context(UserRole.CURATOR),
            patch(
                "tabletop_oracle.api.admin.guardrail_config.GuardrailService"
            ) as mock_service_cls,
        ):
            mock_instance = AsyncMock()
            mock_instance.get_config = AsyncMock(return_value=config)
            mock_service_cls.return_value = mock_instance

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/admin/guardrail-config",
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 200
        body = resp.json()
        data = body["data"]
        assert data["enforcement_enabled"] is True
        assert data["max_tokens_per_query"] == 50000
        assert data["max_model_calls_per_query"] == 10
        assert data["max_queries_per_session"] == 100
        assert data["daily_token_budget"] == 1000000
        assert data["daily_query_budget"] == 5000
        assert data["per_document_ingestion_limit"] == 200000
        assert "meta" in body

    @pytest.mark.asyncio
    async def test_get_returns_config_with_null_limits(self) -> None:
        """Config with null limits serializes correctly."""
        config = _make_config_entity(
            max_tokens_per_query=None,
            max_model_calls_per_query=None,
            max_queries_per_session=None,
            daily_token_budget=None,
            daily_query_budget=None,
            per_document_ingestion_limit=None,
        )

        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with (
            _mock_auth_context(UserRole.CURATOR),
            patch(
                "tabletop_oracle.api.admin.guardrail_config.GuardrailService"
            ) as mock_service_cls,
        ):
            mock_instance = AsyncMock()
            mock_instance.get_config = AsyncMock(return_value=config)
            mock_service_cls.return_value = mock_instance

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/admin/guardrail-config",
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["max_tokens_per_query"] is None
        assert data["daily_token_budget"] is None

    @pytest.mark.asyncio
    async def test_get_returns_404_when_no_config(self) -> None:
        """Returns 404 when no guardrail_config row exists."""
        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with (
            _mock_auth_context(UserRole.CURATOR),
            patch(
                "tabletop_oracle.api.admin.guardrail_config.GuardrailService"
            ) as mock_service_cls,
        ):
            mock_instance = AsyncMock()
            mock_instance.get_config = AsyncMock(
                side_effect=NotFoundError("Guardrail config", "default")
            )
            mock_service_cls.return_value = mock_instance

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/admin/guardrail-config",
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PUT /api/v1/admin/guardrail-config
# ---------------------------------------------------------------------------


class TestUpdateGuardrailConfig:
    """Tests for PUT /api/v1/admin/guardrail-config."""

    @pytest.mark.asyncio
    async def test_update_returns_updated_config(self) -> None:
        """Curator can update config and receives the updated result."""
        updated = _make_config_entity(
            enforcement_enabled=False,
            daily_token_budget=2000000,
        )

        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with (
            _mock_auth_context(UserRole.CURATOR),
            patch(
                "tabletop_oracle.api.admin.guardrail_config.GuardrailService"
            ) as mock_service_cls,
        ):
            mock_instance = AsyncMock()
            mock_instance.update_config = AsyncMock(return_value=updated)
            mock_service_cls.return_value = mock_instance

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.put(
                    "/api/v1/admin/guardrail-config",
                    json={
                        "enforcement_enabled": False,
                        "daily_token_budget": 2000000,
                    },
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["enforcement_enabled"] is False
        assert data["daily_token_budget"] == 2000000

    @pytest.mark.asyncio
    async def test_partial_update_only_sends_provided_fields(self) -> None:
        """Only fields provided in the request body are sent to the service."""
        updated = _make_config_entity(daily_token_budget=2000000)

        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with (
            _mock_auth_context(UserRole.CURATOR),
            patch(
                "tabletop_oracle.api.admin.guardrail_config.GuardrailService"
            ) as mock_service_cls,
        ):
            mock_instance = AsyncMock()
            mock_instance.update_config = AsyncMock(return_value=updated)
            mock_service_cls.return_value = mock_instance

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.put(
                    "/api/v1/admin/guardrail-config",
                    json={"daily_token_budget": 2000000},
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 200
        call_args = mock_instance.update_config.call_args
        updates = call_args[0][0]  # first positional arg
        assert "daily_token_budget" in updates
        assert "enforcement_enabled" not in updates

    @pytest.mark.asyncio
    async def test_update_with_null_value(self) -> None:
        """Setting a field to null in the request body is forwarded."""
        updated = _make_config_entity(max_tokens_per_query=None)

        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with (
            _mock_auth_context(UserRole.CURATOR),
            patch(
                "tabletop_oracle.api.admin.guardrail_config.GuardrailService"
            ) as mock_service_cls,
        ):
            mock_instance = AsyncMock()
            mock_instance.update_config = AsyncMock(return_value=updated)
            mock_service_cls.return_value = mock_instance

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.put(
                    "/api/v1/admin/guardrail-config",
                    json={"max_tokens_per_query": None},
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 200
        call_args = mock_instance.update_config.call_args
        updates = call_args[0][0]
        assert updates["max_tokens_per_query"] is None

    @pytest.mark.asyncio
    async def test_update_returns_404_when_no_config(self) -> None:
        """Returns 404 when no guardrail_config row exists."""
        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with (
            _mock_auth_context(UserRole.CURATOR),
            patch(
                "tabletop_oracle.api.admin.guardrail_config.GuardrailService"
            ) as mock_service_cls,
        ):
            mock_instance = AsyncMock()
            mock_instance.update_config = AsyncMock(
                side_effect=NotFoundError("Guardrail config", "default")
            )
            mock_service_cls.return_value = mock_instance

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.put(
                    "/api/v1/admin/guardrail-config",
                    json={"enforcement_enabled": False},
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Auth enforcement
# ---------------------------------------------------------------------------


class TestAuthEnforcement:
    """Tests for authentication and role enforcement.

    These tests explicitly disable ``bypass_auth`` to ensure the auth
    middleware runs its full validation path, even in CI environments
    where bypass_auth may default to True.
    """

    @pytest.mark.asyncio
    async def test_unauthenticated_returns_401_on_get(self) -> None:
        """Request without session cookie returns 401 on GET."""
        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with patch("tabletop_oracle.auth.middleware.settings") as mock_settings:
            mock_settings.bypass_auth = False
            mock_settings.session_cookie_secure = False
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/v1/admin/guardrail-config")

        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_unauthenticated_returns_401_on_put(self) -> None:
        """Request without session cookie returns 401 on PUT."""
        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with patch("tabletop_oracle.auth.middleware.settings") as mock_settings:
            mock_settings.bypass_auth = False
            mock_settings.session_cookie_secure = False
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.put(
                    "/api/v1/admin/guardrail-config",
                    json={"enforcement_enabled": False},
                )

        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_player_returns_403_on_get(self) -> None:
        """Non-curator (player) returns 403 on GET."""
        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with _mock_auth_context(UserRole.PLAYER):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/admin/guardrail-config",
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_player_returns_403_on_put(self) -> None:
        """Non-curator (player) returns 403 on PUT."""
        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with _mock_auth_context(UserRole.PLAYER):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.put(
                    "/api/v1/admin/guardrail-config",
                    json={"enforcement_enabled": False},
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 403
