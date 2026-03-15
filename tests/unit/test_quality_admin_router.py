"""Unit tests for the quality metrics admin API router."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from tabletop_oracle.api.admin.quality import router as quality_router
from tabletop_oracle.api.deps import get_db
from tabletop_oracle.auth.middleware import SessionMiddleware
from tabletop_oracle.errors.handlers import register_exception_handlers
from tabletop_oracle.middleware.correlation import CorrelationMiddleware
from tabletop_oracle.models.enums import UserRole, UserStatus
from tabletop_oracle.schemas.quality import (
    ConfidenceDistribution,
    GameQualityMetricsResponse,
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
    """Build a minimal FastAPI app with the admin quality router."""
    test_app = FastAPI()
    register_exception_handlers(test_app)
    test_app.add_middleware(SessionMiddleware)
    test_app.add_middleware(CorrelationMiddleware)
    test_app.include_router(quality_router, prefix="/api/v1/admin/quality")
    return test_app


async def _mock_db_session():
    """Dependency override yielding a mock session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    yield session


def _game_quality_metrics() -> list[GameQualityMetricsResponse]:
    """Build test GameQualityMetricsResponse list."""
    return [
        GameQualityMetricsResponse(
            game_id="aaaaaaaa-1111-1111-1111-111111111111",
            game_name="Chess",
            low_confidence_rate=0.15,
            clarification_rate=0.05,
            avg_citations_per_answer=2.3,
            feedback_positive_count=42,
            feedback_negative_count=3,
            confidence_distribution=ConfidenceDistribution(
                low=2,
                medium_low=5,
                medium_high=20,
                high=73,
            ),
            avg_queries_per_session=4.5,
        ),
    ]


class TestGetQualityMetrics:
    """Tests for GET /api/v1/admin/quality/metrics."""

    @pytest.mark.asyncio
    async def test_metrics_returns_per_game_data(self) -> None:
        """Curator receives per-game quality metrics."""
        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with (
            _mock_auth_context(UserRole.CURATOR),
            patch("tabletop_oracle.api.admin.quality.QualityMetricsService") as mock_svc_cls,
        ):
            mock_svc = AsyncMock()
            mock_svc.get_metrics = AsyncMock(return_value=_game_quality_metrics())
            mock_svc_cls.return_value = mock_svc

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/admin/quality/metrics",
                    params={"period_start": "2026-01-01", "period_end": "2026-01-31"},
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data) == 1
        assert data[0]["game_name"] == "Chess"
        assert data[0]["confidence_distribution"]["high"] == 73

    @pytest.mark.asyncio
    async def test_metrics_returns_empty_when_no_games(self) -> None:
        """Metrics returns empty list when no games have activity."""
        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with (
            _mock_auth_context(UserRole.CURATOR),
            patch("tabletop_oracle.api.admin.quality.QualityMetricsService") as mock_svc_cls,
        ):
            mock_svc = AsyncMock()
            mock_svc.get_metrics = AsyncMock(return_value=[])
            mock_svc_cls.return_value = mock_svc

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/admin/quality/metrics",
                    params={"period_start": "2026-01-01", "period_end": "2026-01-31"},
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 200
        assert resp.json()["data"] == []

    @pytest.mark.asyncio
    async def test_metrics_missing_params_returns_error(self) -> None:
        """Missing required query params returns 400."""
        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with _mock_auth_context(UserRole.CURATOR):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/admin/quality/metrics",
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 400


class TestQualityDateRangeValidation:
    """Tests for date range validation on quality endpoints."""

    @pytest.mark.asyncio
    async def test_start_after_end_returns_400(self) -> None:
        """period_start > period_end returns 400."""
        app = _build_test_app()
        app.dependency_overrides[get_db] = _mock_db_session

        with _mock_auth_context(UserRole.CURATOR):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get(
                    "/api/v1/admin/quality/metrics",
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
                    "/api/v1/admin/quality/metrics",
                    params={"period_start": "2024-01-01", "period_end": "2026-01-01"},
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 400


class TestQualityAuthEnforcement:
    """Tests for auth enforcement on quality endpoints."""

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
                    "/api/v1/admin/quality/metrics",
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
                    "/api/v1/admin/quality/metrics",
                    params={"period_start": "2026-01-01", "period_end": "2026-01-31"},
                    cookies={"sid": _VALID_TOKEN},
                )

        assert resp.status_code == 403
