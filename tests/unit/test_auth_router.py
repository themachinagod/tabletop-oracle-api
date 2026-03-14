"""Unit tests for the auth router endpoints.

Tests OAuth login redirect, callback flow, logout, and /me endpoint.
Uses mocked OAuth clients and database sessions to avoid external
dependencies.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from tabletop_oracle.main import app


@pytest.fixture
async def client() -> AsyncClient:
    """Create an async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestLoginEndpoint:
    """Tests for GET /api/v1/auth/login/{provider}."""

    @pytest.mark.asyncio
    async def test_invalid_provider_returns_404(self, client: AsyncClient) -> None:
        """Unknown provider name returns 404."""
        response = await client.get("/api/v1/auth/login/invalid_provider")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_valid_provider_redirects(self, client: AsyncClient) -> None:
        """Valid provider returns a redirect (302) to the auth URL."""
        mock_redirect = MagicMock()
        mock_redirect.status_code = 302
        mock_redirect.headers = {"location": "https://accounts.google.com/o/oauth2/auth?..."}
        mock_redirect.body = b""

        # Build a starlette-like response that has set_cookie
        from starlette.responses import RedirectResponse

        redirect_response = RedirectResponse(
            url="https://accounts.google.com/o/oauth2/auth?state=test",
            status_code=302,
        )

        mock_client = MagicMock()
        mock_client.authorize_redirect = AsyncMock(return_value=redirect_response)

        with patch("tabletop_oracle.api.auth.oauth") as mock_oauth:
            mock_oauth.create_client.return_value = mock_client
            response = await client.get(
                "/api/v1/auth/login/google",
                follow_redirects=False,
            )

        assert response.status_code == 302
        assert "oauth_state" in response.cookies


class TestCallbackEndpoint:
    """Tests for GET /api/v1/auth/callback/{provider}."""

    @pytest.mark.asyncio
    async def test_invalid_provider_returns_404(self, client: AsyncClient) -> None:
        """Unknown provider in callback returns 404."""
        response = await client.get("/api/v1/auth/callback/invalid_provider")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_error_param_returns_401(self, client: AsyncClient) -> None:
        """Provider error parameter returns 401."""
        response = await client.get(
            "/api/v1/auth/callback/google?error=access_denied&error_description=User+denied"
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_state_cookie_returns_401(self, client: AsyncClient) -> None:
        """Missing oauth_state cookie returns 401."""
        response = await client.get("/api/v1/auth/callback/google?state=some-state&code=some-code")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_state_mismatch_returns_401(self, client: AsyncClient) -> None:
        """Mismatched state parameter returns 401."""
        cookies = {"oauth_state": json.dumps({"state": "expected-state"})}
        response = await client.get(
            "/api/v1/auth/callback/google?state=wrong-state&code=some-code",
            cookies=cookies,
        )
        assert response.status_code == 401


class TestLogoutEndpoint:
    """Tests for POST /api/v1/auth/logout."""

    @pytest.mark.asyncio
    async def test_no_session_cookie_returns_401(self, client: AsyncClient) -> None:
        """Missing session cookie returns 401."""
        response = await client.post("/api/v1/auth/logout")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_session_returns_401(self, client: AsyncClient) -> None:
        """Invalid session ID in cookie returns 401."""
        with patch("tabletop_oracle.api.auth.SessionStore") as mock_store:
            mock_store.get = AsyncMock(return_value=None)
            response = await client.post(
                "/api/v1/auth/logout",
                cookies={"sid": "nonexistent-session-id"},
            )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_valid_session_returns_204(self, client: AsyncClient) -> None:
        """Valid session returns 204 and clears cookie."""
        mock_session = MagicMock()
        mock_session.id = "valid-session-id"

        with (
            patch("tabletop_oracle.api.auth.SessionStore") as mock_store,
            patch("tabletop_oracle.api.deps.get_db") as mock_get_db,
        ):
            mock_store.get = AsyncMock(return_value=mock_session)
            mock_store.delete = AsyncMock()

            mock_db = AsyncMock()
            mock_get_db.return_value = mock_db

            async def db_gen():  # type: ignore[no-untyped-def]
                yield mock_db

            mock_get_db.return_value = db_gen()

            response = await client.post(
                "/api/v1/auth/logout",
                cookies={"sid": "valid-session-id"},
            )

        assert response.status_code == 204


class TestMeEndpoint:
    """Tests for GET /api/v1/auth/me."""

    @pytest.mark.asyncio
    async def test_no_session_cookie_returns_401(self, client: AsyncClient) -> None:
        """Missing session cookie returns 401."""
        response = await client.get("/api/v1/auth/me")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_session_returns_401(self, client: AsyncClient) -> None:
        """Invalid session returns 401."""
        with patch("tabletop_oracle.api.auth.SessionStore") as mock_store:
            mock_store.get = AsyncMock(return_value=None)
            response = await client.get(
                "/api/v1/auth/me",
                cookies={"sid": "bad-session"},
            )
        assert response.status_code == 401
