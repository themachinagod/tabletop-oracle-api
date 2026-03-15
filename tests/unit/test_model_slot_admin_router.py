"""Unit tests for the model slot admin API router.

Tests the ``/api/v1/admin/model-slots`` endpoints: listing all slots,
updating a slot, role enforcement, and invalid capability handling.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from tabletop_oracle.api.admin.model_slots import router as model_slots_router
from tabletop_oracle.api.deps import get_db
from tabletop_oracle.auth.middleware import SessionMiddleware
from tabletop_oracle.auth.models import CurrentUser  # noqa: F401
from tabletop_oracle.errors.handlers import register_exception_handlers
from tabletop_oracle.middleware.correlation import CorrelationMiddleware
from tabletop_oracle.models.enums import ModelCapability, UserRole, UserStatus
from tabletop_oracle.services.model.slot_config import ModelSlotConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: Valid 43-char token matching the middleware's session ID pattern.
_VALID_TOKEN = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnop0"


def _make_config(
    capability: ModelCapability = ModelCapability.INTENT_ANALYSIS,
    *,
    provider: str = "openai",
    model_id: str = "gpt-4o",
    max_tokens_per_call: int | None = 4096,
    temperature: float | None = 0.7,
    fallback_provider: str | None = None,
    fallback_model_id: str | None = None,
) -> ModelSlotConfig:
    """Build a ModelSlotConfig for test responses."""
    return ModelSlotConfig(
        capability=capability,
        provider=provider,
        model_id=model_id,
        max_tokens_per_call=max_tokens_per_call,
        temperature=temperature,
        fallback_provider=fallback_provider,
        fallback_model_id=fallback_model_id,
    )


def _make_mock_session(
    user_id: uuid.UUID,
    expired: bool = False,
) -> MagicMock:
    """Create a mock AuthSession object."""
    from datetime import UTC, datetime, timedelta

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
    """Build a minimal FastAPI app with admin model-slots router."""
    test_app = FastAPI()
    register_exception_handlers(test_app)
    test_app.add_middleware(SessionMiddleware)
    test_app.add_middleware(CorrelationMiddleware)
    test_app.include_router(model_slots_router, prefix="/api/v1/admin/model-slots")
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
# List all slots
# ---------------------------------------------------------------------------


class TestListModelSlots:
    """Tests for GET /api/v1/admin/model-slots."""

    @pytest.mark.asyncio
    async def test_list_returns_all_slots(self) -> None:
        """Curator can list all model slots."""
        configs = [
            _make_config(cap)
            for cap in [
                ModelCapability.INTENT_ANALYSIS,
                ModelCapability.ANSWER_SYNTHESIS,
                ModelCapability.CONCEPT_EXTRACTION,
            ]
        ]

        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with (
            _mock_auth_context(UserRole.CURATOR),
            patch("tabletop_oracle.api.admin.model_slots.ModelSlotService") as mock_service_cls,
        ):
            mock_instance = AsyncMock()
            mock_instance.list_all = AsyncMock(return_value=configs)
            mock_service_cls.return_value = mock_instance

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/admin/model-slots",
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["data"]) == 3
        assert body["data"][0]["capability"] == "intent_analysis"

    @pytest.mark.asyncio
    async def test_list_returns_empty_when_no_slots(self) -> None:
        """List endpoint returns empty array when no slots exist."""
        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with (
            _mock_auth_context(UserRole.CURATOR),
            patch("tabletop_oracle.api.admin.model_slots.ModelSlotService") as mock_service_cls,
        ):
            mock_instance = AsyncMock()
            mock_instance.list_all = AsyncMock(return_value=[])
            mock_service_cls.return_value = mock_instance

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/admin/model-slots",
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 200
        assert resp.json()["data"] == []


# ---------------------------------------------------------------------------
# Update slot
# ---------------------------------------------------------------------------


class TestUpdateModelSlot:
    """Tests for PUT /api/v1/admin/model-slots/{capability}."""

    @pytest.mark.asyncio
    async def test_update_slot_returns_updated_config(self) -> None:
        """Curator can update a slot and receives the updated config."""
        updated = _make_config(
            ModelCapability.INTENT_ANALYSIS,
            provider="anthropic",
            model_id="claude-sonnet-4-20250514",
        )

        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with (
            _mock_auth_context(UserRole.CURATOR),
            patch("tabletop_oracle.api.admin.model_slots.ModelSlotService") as mock_service_cls,
        ):
            mock_instance = AsyncMock()
            mock_instance.update_slot = AsyncMock(return_value=updated)
            mock_service_cls.return_value = mock_instance

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.put(
                    "/api/v1/admin/model-slots/intent_analysis",
                    json={
                        "provider": "anthropic",
                        "model_id": "claude-sonnet-4-20250514",
                    },
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["provider"] == "anthropic"
        assert data["model_id"] == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_partial_update_only_sends_provided_fields(self) -> None:
        """Only fields provided in the request body are sent to the service."""
        updated = _make_config(
            ModelCapability.ANSWER_SYNTHESIS,
            temperature=0.5,
        )

        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with (
            _mock_auth_context(UserRole.CURATOR),
            patch("tabletop_oracle.api.admin.model_slots.ModelSlotService") as mock_service_cls,
        ):
            mock_instance = AsyncMock()
            mock_instance.update_slot = AsyncMock(return_value=updated)
            mock_service_cls.return_value = mock_instance

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.put(
                    "/api/v1/admin/model-slots/answer_synthesis",
                    json={"temperature": 0.5},
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 200
        # Verify only temperature was in the update dict
        call_args = mock_instance.update_slot.call_args
        updates = call_args[0][1]  # second positional arg
        assert "temperature" in updates
        assert "provider" not in updates


# ---------------------------------------------------------------------------
# Invalid capability (404)
# ---------------------------------------------------------------------------


class TestInvalidCapability:
    """Tests for 404 on invalid capability path parameter."""

    @pytest.mark.asyncio
    async def test_invalid_capability_returns_404(self) -> None:
        """PUT with an invalid capability returns 404."""
        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with _mock_auth_context(UserRole.CURATOR):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.put(
                    "/api/v1/admin/model-slots/nonexistent_capability",
                    json={"provider": "openai"},
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
    async def test_unauthenticated_returns_401(self) -> None:
        """Request without session cookie returns 401."""
        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with patch("tabletop_oracle.auth.middleware.settings") as mock_settings:
            mock_settings.bypass_auth = False
            mock_settings.session_cookie_secure = False
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/v1/admin/model-slots")

        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_player_returns_403_on_list(self) -> None:
        """Non-curator (player) returns 403 on GET."""
        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with _mock_auth_context(UserRole.PLAYER):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/admin/model-slots",
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_player_returns_403_on_update(self) -> None:
        """Non-curator (player) returns 403 on PUT."""
        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with _mock_auth_context(UserRole.PLAYER):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.put(
                    "/api/v1/admin/model-slots/intent_analysis",
                    json={"provider": "anthropic"},
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 403
