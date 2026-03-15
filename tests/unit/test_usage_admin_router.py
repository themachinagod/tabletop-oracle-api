"""Unit tests for the usage aggregation admin API router."""

from __future__ import annotations

import uuid
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from tabletop_oracle.api.admin.usage import router as usage_router
from tabletop_oracle.api.deps import get_db
from tabletop_oracle.auth.middleware import SessionMiddleware
from tabletop_oracle.errors.handlers import register_exception_handlers
from tabletop_oracle.middleware.correlation import CorrelationMiddleware
from tabletop_oracle.models.enums import UserRole, UserStatus
from tabletop_oracle.schemas.usage import (
    CapabilityUsageResponse,
    DailyUsageResponse,
    GameUsageResponse,
    GuardrailIndicator,
    GuardrailStatusResponse,
    GuardrailThreshold,
    UsageSummaryResponse,
    UserUsageResponse,
)

_VALID_TOKEN = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnop0"
_USER_ID = uuid.UUID("12345678-1234-1234-1234-123456789abc")


def _make_mock_session(user_id: uuid.UUID) -> MagicMock:
    """Create a mock AuthSession object."""
    from datetime import UTC, datetime, timedelta

    session = MagicMock()
    session.user_id = user_id
    session.expires_at = datetime.now(UTC) + timedelta(days=30)
    return session


def _mock_auth_context(role: UserRole = UserRole.CURATOR):
    """Context manager patching auth middleware for a user with given role."""
    mock_session = _make_mock_session(user_id=_USER_ID)
    mock_user = MagicMock()
    mock_user.id = _USER_ID
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
            self._settings_patch = patch("tabletop_oracle.auth.middleware.settings")
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
    """Build a minimal FastAPI app with the admin usage router."""
    test_app = FastAPI()
    register_exception_handlers(test_app)
    test_app.add_middleware(SessionMiddleware)
    test_app.add_middleware(CorrelationMiddleware)
    test_app.include_router(usage_router, prefix="/api/v1/admin/usage")
    return test_app


async def _mock_db_session():
    """Dependency override yielding a mock session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    yield session


def _summary_response() -> UsageSummaryResponse:
    """Build a test UsageSummaryResponse."""
    return UsageSummaryResponse(
        total_tokens=100000,
        input_tokens=60000,
        output_tokens=40000,
        total_queries=150,
        total_documents_processed=10,
        unique_users=5,
        period_start=date(2026, 1, 1),
        period_end=date(2026, 1, 31),
    )


class TestGetUsageSummary:
    """Tests for GET /api/v1/admin/usage/summary."""

    @pytest.mark.asyncio
    async def test_summary_returns_data(self) -> None:
        """Curator receives usage summary for a valid date range."""
        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with (
            _mock_auth_context(UserRole.CURATOR),
            patch("tabletop_oracle.api.admin.usage.UsageAggregationService") as mock_svc_cls,
        ):
            mock_svc = AsyncMock()
            mock_svc.get_summary = AsyncMock(return_value=_summary_response())
            mock_svc_cls.return_value = mock_svc

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/admin/usage/summary",
                    params={"period_start": "2026-01-01", "period_end": "2026-01-31"},
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["total_tokens"] == 100000

    @pytest.mark.asyncio
    async def test_summary_missing_params_returns_error(self) -> None:
        """Missing required query params returns 400."""
        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with _mock_auth_context(UserRole.CURATOR):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/admin/usage/summary",
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 400


class TestGetUsageTrends:
    """Tests for GET /api/v1/admin/usage/trends."""

    @pytest.mark.asyncio
    async def test_trends_returns_daily_data(self) -> None:
        """Curator receives daily usage trend data."""
        trends = [
            DailyUsageResponse(
                date=date(2026, 1, 1), total_tokens=5000, query_count=10, document_count=1
            ),
            DailyUsageResponse(
                date=date(2026, 1, 2), total_tokens=7000, query_count=15, document_count=2
            ),
        ]
        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with (
            _mock_auth_context(UserRole.CURATOR),
            patch("tabletop_oracle.api.admin.usage.UsageAggregationService") as mock_svc_cls,
        ):
            mock_svc = AsyncMock()
            mock_svc.get_trends = AsyncMock(return_value=trends)
            mock_svc_cls.return_value = mock_svc

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/admin/usage/trends",
                    params={"period_start": "2026-01-01", "period_end": "2026-01-02"},
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 200
        assert len(resp.json()["data"]) == 2


class TestGetUsageByCapability:
    """Tests for GET /api/v1/admin/usage/by-capability."""

    @pytest.mark.asyncio
    async def test_by_capability_returns_breakdown(self) -> None:
        """Curator receives capability breakdown."""
        caps = [
            CapabilityUsageResponse(
                capability="intent_analysis",
                total_tokens=30000,
                call_count=50,
                input_tokens=20000,
                output_tokens=10000,
            ),
        ]
        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with (
            _mock_auth_context(UserRole.CURATOR),
            patch("tabletop_oracle.api.admin.usage.UsageAggregationService") as mock_svc_cls,
        ):
            mock_svc = AsyncMock()
            mock_svc.get_by_capability = AsyncMock(return_value=caps)
            mock_svc_cls.return_value = mock_svc

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/admin/usage/by-capability",
                    params={"period_start": "2026-01-01", "period_end": "2026-01-31"},
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 200
        assert resp.json()["data"][0]["capability"] == "intent_analysis"


class TestGetUsageByGame:
    """Tests for GET /api/v1/admin/usage/by-game."""

    @pytest.mark.asyncio
    async def test_by_game_returns_breakdown(self) -> None:
        """Curator receives game breakdown."""
        games = [
            GameUsageResponse(
                game_id="aaaaaaaa-1111-1111-1111-111111111111",
                game_name="Chess",
                query_tokens=20000,
                ingestion_tokens=5000,
                query_count=30,
            ),
        ]
        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with (
            _mock_auth_context(UserRole.CURATOR),
            patch("tabletop_oracle.api.admin.usage.UsageAggregationService") as mock_svc_cls,
        ):
            mock_svc = AsyncMock()
            mock_svc.get_by_game = AsyncMock(return_value=games)
            mock_svc_cls.return_value = mock_svc

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/admin/usage/by-game",
                    params={"period_start": "2026-01-01", "period_end": "2026-01-31"},
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 200
        assert resp.json()["data"][0]["game_name"] == "Chess"


class TestGetUsageByUser:
    """Tests for GET /api/v1/admin/usage/by-user."""

    @pytest.mark.asyncio
    async def test_by_user_returns_breakdown(self) -> None:
        """Curator receives user breakdown."""
        users = [
            UserUsageResponse(
                user_id="bbbbbbbb-2222-2222-2222-222222222222",
                display_name="Alice",
                total_tokens=15000,
                query_count=20,
            ),
        ]
        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with (
            _mock_auth_context(UserRole.CURATOR),
            patch("tabletop_oracle.api.admin.usage.UsageAggregationService") as mock_svc_cls,
        ):
            mock_svc = AsyncMock()
            mock_svc.get_by_user = AsyncMock(return_value=users)
            mock_svc_cls.return_value = mock_svc

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/admin/usage/by-user",
                    params={"period_start": "2026-01-01", "period_end": "2026-01-31"},
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 200
        assert resp.json()["data"][0]["display_name"] == "Alice"


class TestGetGuardrailStatus:
    """Tests for GET /api/v1/admin/usage/guardrail-status."""

    @pytest.mark.asyncio
    async def test_guardrail_status_returns_indicators(self) -> None:
        """Curator receives guardrail status indicators."""
        status = GuardrailStatusResponse(
            enforcement_enabled=True,
            indicators=[
                GuardrailIndicator(
                    name="Daily Token Budget",
                    current=45000,
                    limit=100000,
                    status=GuardrailThreshold.GREEN,
                ),
                GuardrailIndicator(
                    name="Daily Query Budget",
                    current=4500,
                    limit=5000,
                    status=GuardrailThreshold.RED,
                ),
            ],
        )
        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with (
            _mock_auth_context(UserRole.CURATOR),
            patch("tabletop_oracle.api.admin.usage.UsageAggregationService") as mock_svc_cls,
        ):
            mock_svc = AsyncMock()
            mock_svc.get_guardrail_status = AsyncMock(return_value=status)
            mock_svc_cls.return_value = mock_svc

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/admin/usage/guardrail-status",
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["enforcement_enabled"] is True
        assert len(data["indicators"]) == 2

    @pytest.mark.asyncio
    async def test_guardrail_status_no_date_params_required(self) -> None:
        """Guardrail status does not require date range params."""
        status = GuardrailStatusResponse(enforcement_enabled=False, indicators=[])
        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with (
            _mock_auth_context(UserRole.CURATOR),
            patch("tabletop_oracle.api.admin.usage.UsageAggregationService") as mock_svc_cls,
        ):
            mock_svc = AsyncMock()
            mock_svc.get_guardrail_status = AsyncMock(return_value=status)
            mock_svc_cls.return_value = mock_svc

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/admin/usage/guardrail-status",
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 200


class TestDateRangeValidation:
    """Tests for date range validation across usage endpoints."""

    @pytest.mark.asyncio
    async def test_start_after_end_returns_400(self) -> None:
        """period_start > period_end returns 400."""
        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with _mock_auth_context(UserRole.CURATOR):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/admin/usage/summary",
                    params={"period_start": "2026-02-01", "period_end": "2026-01-01"},
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_range_exceeds_365_days_returns_400(self) -> None:
        """Date range > 365 days returns 400."""
        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with _mock_auth_context(UserRole.CURATOR):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/admin/usage/summary",
                    params={"period_start": "2024-01-01", "period_end": "2026-01-01"},
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_same_day_range_is_valid(self) -> None:
        """period_start == period_end is valid (single day)."""
        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with (
            _mock_auth_context(UserRole.CURATOR),
            patch("tabletop_oracle.api.admin.usage.UsageAggregationService") as mock_svc_cls,
        ):
            mock_svc = AsyncMock()
            mock_svc.get_summary = AsyncMock(return_value=_summary_response())
            mock_svc_cls.return_value = mock_svc

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/admin/usage/summary",
                    params={"period_start": "2026-01-15", "period_end": "2026-01-15"},
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 200


class TestUsageAuthEnforcement:
    """Tests for auth enforcement on usage endpoints."""

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
                resp = await client.get(
                    "/api/v1/admin/usage/summary",
                    params={"period_start": "2026-01-01", "period_end": "2026-01-31"},
                )

        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_player_returns_403(self) -> None:
        """Non-curator (player) returns 403."""
        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with _mock_auth_context(UserRole.PLAYER):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/admin/usage/summary",
                    params={"period_start": "2026-01-01", "period_end": "2026-01-31"},
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 403
