"""Integration tests for expansion API endpoints against a real PostgreSQL database.

Exercises all 6 expansion endpoints (create, list, detail, update,
archive, restore) through the full HTTP stack with real database
operations, plus game silo isolation tests. Auth is handled via
bypass_auth mode (curator role).
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from tabletop_oracle.api.deps import get_db
from tabletop_oracle.main import app

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.ext.asyncio import AsyncSession

_REQUEST_ID = "test-expansion-endpoints"
_BASE_URL = "/api/v1/games"


@pytest.fixture
async def _seed_user(db_session: AsyncSession) -> uuid.UUID:
    """Create a curator user via raw SQL for game/expansion creation attribution."""
    user_id = uuid.UUID("00000000-0000-0000-0000-000000000000")
    await db_session.execute(
        text(
            "INSERT INTO users (id, oauth_provider, oauth_subject_id, email, display_name, role) "
            "VALUES (:id, :provider, :sub, :email, :name, :role)"
        ),
        {
            "id": str(user_id),
            "provider": "google",
            "sub": "bypass-subject",
            "email": "dev@localhost",
            "name": "Dev Bypass",
            "role": "curator",
        },
    )
    return user_id


@pytest.fixture
async def api_client(
    db_session: AsyncSession,
    _seed_user: uuid.UUID,
) -> AsyncGenerator[AsyncClient, None]:
    """Provide an HTTP client with real DB and bypass auth enabled."""

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    try:
        with patch("tabletop_oracle.auth.middleware.settings") as mock_settings:
            mock_settings.bypass_auth = True
            mock_settings.session_cookie_secure = False
            mock_settings.auth_session_timeout_days = 30
            mock_settings.frontend_origin = "http://localhost:4200"
            mock_settings.log_level = "DEBUG"
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport,
                base_url="http://test",
            ) as client:
                yield client
    finally:
        app.dependency_overrides.clear()


async def _create_game(client: AsyncClient, name: str = "Test Game") -> str:
    """Helper to create a game and return its ID."""
    resp = await client.post(f"{_BASE_URL}", json={"name": name})
    assert resp.status_code == 201
    return resp.json()["data"]["id"]


def _expansion_payload(**overrides: Any) -> dict[str, Any]:
    """Build a minimal valid expansion creation payload with optional overrides."""
    base: dict[str, Any] = {"name": "Seafarers"}
    base.update(overrides)
    return base


@pytest.mark.integration
class TestCreateExpansion:
    """POST /api/v1/games/{game_id}/expansions -- create expansion."""

    async def test_create_expansion_returns_201_with_location(
        self, api_client: AsyncClient
    ) -> None:
        """Successful creation returns 201, Location header, and envelope."""
        game_id = await _create_game(api_client)
        payload = _expansion_payload(
            year_published=1997,
            description="Explore and settle new islands.",
        )

        response = await api_client.post(
            f"{_BASE_URL}/{game_id}/expansions",
            json=payload,
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 201
        body = response.json()

        # Envelope structure
        assert "data" in body
        assert "meta" in body
        assert body["meta"]["request_id"] == _REQUEST_ID

        # Expansion data
        data = body["data"]
        assert data["name"] == "Seafarers"
        assert data["year_published"] == 1997
        assert data["description"] == "Explore and settle new islands."
        assert data["game_id"] == game_id
        assert data["archived_at"] is None

        # Location header
        assert "location" in response.headers
        assert data["id"] in response.headers["location"]
        assert game_id in response.headers["location"]

    async def test_create_expansion_minimal_payload(self, api_client: AsyncClient) -> None:
        """Minimal valid payload (name only) returns 201."""
        game_id = await _create_game(api_client)
        response = await api_client.post(
            f"{_BASE_URL}/{game_id}/expansions",
            json=_expansion_payload(),
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 201
        data = response.json()["data"]
        assert data["name"] == "Seafarers"
        assert data["year_published"] is None
        assert data["description"] is None

    async def test_create_expansion_missing_name_returns_400(
        self, api_client: AsyncClient
    ) -> None:
        """Missing required name field returns 400."""
        game_id = await _create_game(api_client)
        response = await api_client.post(
            f"{_BASE_URL}/{game_id}/expansions",
            json={},
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 400

    async def test_create_expansion_nonexistent_game_returns_404(
        self, api_client: AsyncClient
    ) -> None:
        """Creating expansion under non-existent game returns 404."""
        fake_game_id = str(uuid.uuid4())
        response = await api_client.post(
            f"{_BASE_URL}/{fake_game_id}/expansions",
            json=_expansion_payload(),
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 404


@pytest.mark.integration
class TestListExpansions:
    """GET /api/v1/games/{game_id}/expansions -- list with filtering and pagination."""

    async def _create_expansion(
        self, client: AsyncClient, game_id: str, name: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Helper to create an expansion and return its data."""
        payload = _expansion_payload(name=name, **kwargs)
        resp = await client.post(f"{_BASE_URL}/{game_id}/expansions", json=payload)
        assert resp.status_code == 201
        return resp.json()["data"]

    async def test_list_expansions_returns_paginated_envelope(
        self, api_client: AsyncClient
    ) -> None:
        """List returns ListEnvelope with pagination metadata."""
        game_id = await _create_game(api_client)
        await self._create_expansion(api_client, game_id, "Expansion A")
        await self._create_expansion(api_client, game_id, "Expansion B")

        response = await api_client.get(
            f"{_BASE_URL}/{game_id}/expansions",
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 200
        body = response.json()
        assert "data" in body
        assert "meta" in body
        assert body["meta"]["request_id"] == _REQUEST_ID
        assert "pagination" in body["meta"]

        pagination = body["meta"]["pagination"]
        assert pagination["total_count"] == 2
        assert pagination["page"] == 1

    async def test_list_expansions_includes_document_count(self, api_client: AsyncClient) -> None:
        """List response includes document_count field per expansion."""
        game_id = await _create_game(api_client)
        await self._create_expansion(api_client, game_id, "With Doc Count")

        response = await api_client.get(
            f"{_BASE_URL}/{game_id}/expansions",
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert len(data) == 1
        assert "document_count" in data[0]
        assert data[0]["document_count"] == 0

    async def test_list_expansions_excludes_archived_by_default(
        self, api_client: AsyncClient
    ) -> None:
        """Archived expansions are excluded when archived=false (default)."""
        game_id = await _create_game(api_client)
        exp = await self._create_expansion(api_client, game_id, "To Archive")
        await api_client.post(f"{_BASE_URL}/{game_id}/expansions/{exp['id']}/archive")

        response = await api_client.get(
            f"{_BASE_URL}/{game_id}/expansions",
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 200
        names = [e["name"] for e in response.json()["data"]]
        assert "To Archive" not in names

    async def test_list_expansions_includes_archived_when_requested(
        self, api_client: AsyncClient
    ) -> None:
        """Archived expansions included when archived=true."""
        game_id = await _create_game(api_client)
        exp = await self._create_expansion(api_client, game_id, "Archived Visible")
        await api_client.post(f"{_BASE_URL}/{game_id}/expansions/{exp['id']}/archive")

        response = await api_client.get(
            f"{_BASE_URL}/{game_id}/expansions",
            params={"archived": "true"},
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 200
        names = [e["name"] for e in response.json()["data"]]
        assert "Archived Visible" in names

    async def test_list_expansions_pagination(self, api_client: AsyncClient) -> None:
        """Pagination parameters control page size and offset."""
        game_id = await _create_game(api_client)
        for i in range(5):
            await self._create_expansion(api_client, game_id, f"Exp {i}")

        response = await api_client.get(
            f"{_BASE_URL}/{game_id}/expansions",
            params={"page": 1, "page_size": 2},
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 200
        body = response.json()
        assert len(body["data"]) == 2
        assert body["meta"]["pagination"]["page_size"] == 2
        assert body["meta"]["pagination"]["total_count"] == 5

    async def test_list_expansions_nonexistent_game_returns_404(
        self, api_client: AsyncClient
    ) -> None:
        """Listing expansions for a non-existent game returns 404."""
        fake_game_id = str(uuid.uuid4())
        response = await api_client.get(
            f"{_BASE_URL}/{fake_game_id}/expansions",
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 404


@pytest.mark.integration
class TestGetExpansionDetail:
    """GET /api/v1/games/{game_id}/expansions/{expansion_id} -- expansion detail."""

    async def test_get_expansion_returns_detail_envelope(self, api_client: AsyncClient) -> None:
        """Detail endpoint returns ExpansionResponse in envelope."""
        game_id = await _create_game(api_client)
        create_resp = await api_client.post(
            f"{_BASE_URL}/{game_id}/expansions",
            json=_expansion_payload(
                name="Detail Expansion",
                year_published=2020,
                description="A detailed expansion.",
            ),
        )
        expansion_id = create_resp.json()["data"]["id"]

        response = await api_client.get(
            f"{_BASE_URL}/{game_id}/expansions/{expansion_id}",
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["meta"]["request_id"] == _REQUEST_ID

        data = body["data"]
        assert data["id"] == expansion_id
        assert data["game_id"] == game_id
        assert data["name"] == "Detail Expansion"
        assert data["year_published"] == 2020

    async def test_get_expansion_not_found(self, api_client: AsyncClient) -> None:
        """Non-existent expansion returns 404."""
        game_id = await _create_game(api_client)
        fake_id = str(uuid.uuid4())
        response = await api_client.get(
            f"{_BASE_URL}/{game_id}/expansions/{fake_id}",
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 404


@pytest.mark.integration
class TestUpdateExpansion:
    """PATCH /api/v1/games/{game_id}/expansions/{expansion_id} -- partial update."""

    async def test_update_expansion_partial(self, api_client: AsyncClient) -> None:
        """Partial update modifies only provided fields."""
        game_id = await _create_game(api_client)
        create_resp = await api_client.post(
            f"{_BASE_URL}/{game_id}/expansions",
            json=_expansion_payload(name="Before Update", description="Original"),
        )
        expansion_id = create_resp.json()["data"]["id"]

        response = await api_client.patch(
            f"{_BASE_URL}/{game_id}/expansions/{expansion_id}",
            json={"description": "Updated Description"},
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["description"] == "Updated Description"
        assert data["name"] == "Before Update"

    async def test_update_expansion_name(self, api_client: AsyncClient) -> None:
        """Name can be updated via PATCH."""
        game_id = await _create_game(api_client)
        create_resp = await api_client.post(
            f"{_BASE_URL}/{game_id}/expansions",
            json=_expansion_payload(name="Old Name"),
        )
        expansion_id = create_resp.json()["data"]["id"]

        response = await api_client.patch(
            f"{_BASE_URL}/{game_id}/expansions/{expansion_id}",
            json={"name": "New Name"},
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 200
        assert response.json()["data"]["name"] == "New Name"

    async def test_update_expansion_not_found(self, api_client: AsyncClient) -> None:
        """Updating non-existent expansion returns 404."""
        game_id = await _create_game(api_client)
        fake_id = str(uuid.uuid4())
        response = await api_client.patch(
            f"{_BASE_URL}/{game_id}/expansions/{fake_id}",
            json={"name": "Nope"},
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 404


@pytest.mark.integration
class TestArchiveExpansion:
    """POST .../archive -- soft delete."""

    async def test_archive_expansion_success(self, api_client: AsyncClient) -> None:
        """Archive sets archived_at and returns the expansion."""
        game_id = await _create_game(api_client)
        create_resp = await api_client.post(
            f"{_BASE_URL}/{game_id}/expansions",
            json=_expansion_payload(name="Archivable"),
        )
        expansion_id = create_resp.json()["data"]["id"]

        response = await api_client.post(
            f"{_BASE_URL}/{game_id}/expansions/{expansion_id}/archive",
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["archived_at"] is not None

    async def test_archive_already_archived_returns_409(self, api_client: AsyncClient) -> None:
        """Archiving an already-archived expansion returns 409."""
        game_id = await _create_game(api_client)
        create_resp = await api_client.post(
            f"{_BASE_URL}/{game_id}/expansions",
            json=_expansion_payload(name="Double Archive"),
        )
        expansion_id = create_resp.json()["data"]["id"]

        await api_client.post(f"{_BASE_URL}/{game_id}/expansions/{expansion_id}/archive")
        response = await api_client.post(
            f"{_BASE_URL}/{game_id}/expansions/{expansion_id}/archive",
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 409

    async def test_archive_nonexistent_expansion_returns_404(
        self, api_client: AsyncClient
    ) -> None:
        """Archiving non-existent expansion returns 404."""
        game_id = await _create_game(api_client)
        fake_id = str(uuid.uuid4())
        response = await api_client.post(
            f"{_BASE_URL}/{game_id}/expansions/{fake_id}/archive",
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 404


@pytest.mark.integration
class TestRestoreExpansion:
    """POST .../restore -- un-archive."""

    async def test_restore_expansion_success(self, api_client: AsyncClient) -> None:
        """Restore clears archived_at and returns the expansion."""
        game_id = await _create_game(api_client)
        create_resp = await api_client.post(
            f"{_BASE_URL}/{game_id}/expansions",
            json=_expansion_payload(name="Restorable"),
        )
        expansion_id = create_resp.json()["data"]["id"]

        await api_client.post(f"{_BASE_URL}/{game_id}/expansions/{expansion_id}/archive")
        response = await api_client.post(
            f"{_BASE_URL}/{game_id}/expansions/{expansion_id}/restore",
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["archived_at"] is None

    async def test_restore_not_archived_returns_409(self, api_client: AsyncClient) -> None:
        """Restoring a non-archived expansion returns 409."""
        game_id = await _create_game(api_client)
        create_resp = await api_client.post(
            f"{_BASE_URL}/{game_id}/expansions",
            json=_expansion_payload(name="Not Archived"),
        )
        expansion_id = create_resp.json()["data"]["id"]

        response = await api_client.post(
            f"{_BASE_URL}/{game_id}/expansions/{expansion_id}/restore",
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 409


@pytest.mark.integration
class TestGameSiloIsolation:
    """Verify game silo isolation -- expansions are only accessible via their owning game."""

    async def test_get_expansion_via_wrong_game_returns_404(self, api_client: AsyncClient) -> None:
        """Accessing expansion via a different game_id returns 404."""
        game_a = await _create_game(api_client, "Game A")
        game_b = await _create_game(api_client, "Game B")

        create_resp = await api_client.post(
            f"{_BASE_URL}/{game_a}/expansions",
            json=_expansion_payload(name="Game A Expansion"),
        )
        expansion_id = create_resp.json()["data"]["id"]

        # Access via correct game — should work
        response = await api_client.get(
            f"{_BASE_URL}/{game_a}/expansions/{expansion_id}",
            headers={"X-Request-ID": _REQUEST_ID},
        )
        assert response.status_code == 200

        # Access via wrong game — should 404
        response = await api_client.get(
            f"{_BASE_URL}/{game_b}/expansions/{expansion_id}",
            headers={"X-Request-ID": _REQUEST_ID},
        )
        assert response.status_code == 404

    async def test_update_expansion_via_wrong_game_returns_404(
        self, api_client: AsyncClient
    ) -> None:
        """Updating expansion via a different game_id returns 404."""
        game_a = await _create_game(api_client, "Game A")
        game_b = await _create_game(api_client, "Game B")

        create_resp = await api_client.post(
            f"{_BASE_URL}/{game_a}/expansions",
            json=_expansion_payload(name="Silo Test"),
        )
        expansion_id = create_resp.json()["data"]["id"]

        response = await api_client.patch(
            f"{_BASE_URL}/{game_b}/expansions/{expansion_id}",
            json={"name": "Hacked"},
            headers={"X-Request-ID": _REQUEST_ID},
        )
        assert response.status_code == 404

    async def test_archive_expansion_via_wrong_game_returns_404(
        self, api_client: AsyncClient
    ) -> None:
        """Archiving expansion via a different game_id returns 404."""
        game_a = await _create_game(api_client, "Game A")
        game_b = await _create_game(api_client, "Game B")

        create_resp = await api_client.post(
            f"{_BASE_URL}/{game_a}/expansions",
            json=_expansion_payload(name="Silo Archive"),
        )
        expansion_id = create_resp.json()["data"]["id"]

        response = await api_client.post(
            f"{_BASE_URL}/{game_b}/expansions/{expansion_id}/archive",
            headers={"X-Request-ID": _REQUEST_ID},
        )
        assert response.status_code == 404

    async def test_list_expansions_only_shows_own_game(self, api_client: AsyncClient) -> None:
        """List only returns expansions belonging to the specified game."""
        game_a = await _create_game(api_client, "Game A")
        game_b = await _create_game(api_client, "Game B")

        await api_client.post(
            f"{_BASE_URL}/{game_a}/expansions",
            json=_expansion_payload(name="A Expansion"),
        )
        await api_client.post(
            f"{_BASE_URL}/{game_b}/expansions",
            json=_expansion_payload(name="B Expansion"),
        )

        response_a = await api_client.get(
            f"{_BASE_URL}/{game_a}/expansions",
            headers={"X-Request-ID": _REQUEST_ID},
        )
        response_b = await api_client.get(
            f"{_BASE_URL}/{game_b}/expansions",
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response_a.status_code == 200
        assert response_b.status_code == 200

        names_a = [e["name"] for e in response_a.json()["data"]]
        names_b = [e["name"] for e in response_b.json()["data"]]

        assert "A Expansion" in names_a
        assert "B Expansion" not in names_a
        assert "B Expansion" in names_b
        assert "A Expansion" not in names_b


@pytest.mark.integration
class TestAuthEnforcement:
    """Verify authentication enforcement on expansion endpoints."""

    async def test_unauthenticated_request_returns_401(
        self, db_session: AsyncSession, _seed_user: uuid.UUID
    ) -> None:
        """Request without auth returns 401."""

        async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
            yield db_session

        app.dependency_overrides[get_db] = _override_get_db
        try:
            with patch("tabletop_oracle.auth.middleware.settings") as mock_settings:
                mock_settings.bypass_auth = False
                mock_settings.session_cookie_secure = False
                mock_settings.auth_session_timeout_days = 30
                mock_settings.frontend_origin = "http://localhost:4200"
                mock_settings.log_level = "DEBUG"
                transport = ASGITransport(app=app)
                async with AsyncClient(
                    transport=transport,
                    base_url="http://test",
                ) as client:
                    fake_game_id = str(uuid.uuid4())
                    response = await client.get(
                        f"{_BASE_URL}/{fake_game_id}/expansions",
                        headers={"X-Request-ID": _REQUEST_ID},
                    )

                assert response.status_code == 401
        finally:
            app.dependency_overrides.clear()
