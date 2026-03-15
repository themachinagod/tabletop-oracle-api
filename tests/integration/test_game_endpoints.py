"""Integration tests for game API endpoints against a real PostgreSQL database.

Exercises all 7 game endpoints (create, list, tags, detail, update,
archive, restore) through the full HTTP stack with real database
operations. Auth is handled via bypass_auth mode (curator role).
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from tabletop_oracle.api.deps import get_db
from tabletop_oracle.main import app
from tabletop_oracle.models.enums import UserRole, UserStatus
from tabletop_oracle.models.user import User

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.ext.asyncio import AsyncSession

_REQUEST_ID = "test-game-endpoints"


@pytest.fixture
async def _seed_user(db_session: AsyncSession) -> User:
    """Create a curator user for game creation attribution."""
    user = User(
        id=uuid.UUID("00000000-0000-0000-0000-000000000000"),
        email="dev@localhost",
        display_name="Dev Bypass",
        role=UserRole.CURATOR,
        status=UserStatus.ACTIVE,
        oauth_provider="google",
        oauth_subject_id="bypass-subject",
    )
    db_session.add(user)
    await db_session.flush()
    return user


@pytest.fixture
async def api_client(
    db_session: AsyncSession,
    _seed_user: User,
) -> AsyncGenerator[AsyncClient, None]:
    """Provide an HTTP client with real DB and bypass auth enabled.

    Overrides the get_db dependency to use the test session (with
    savepoint isolation) and enables bypass_auth for curator access.
    """

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


def _game_payload(**overrides: Any) -> dict[str, Any]:
    """Build a minimal valid game creation payload with optional overrides."""
    base: dict[str, Any] = {"name": "Catan"}
    base.update(overrides)
    return base


@pytest.mark.integration
class TestCreateGame:
    """POST /api/v1/games — create game."""

    async def test_create_game_returns_201_with_location(self, api_client: AsyncClient) -> None:
        """Successful creation returns 201, Location header, and envelope."""
        payload = _game_payload(
            publisher="Catan Studio",
            year_published=1995,
            min_players=3,
            max_players=4,
            complexity="medium",
            tags=["strategy", "trading"],
        )
        response = await api_client.post(
            "/api/v1/games",
            json=payload,
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 201
        body = response.json()

        # Envelope structure
        assert "data" in body
        assert "meta" in body
        assert body["meta"]["request_id"] == _REQUEST_ID

        # Game data
        data = body["data"]
        assert data["name"] == "Catan"
        assert data["publisher"] == "Catan Studio"
        assert data["complexity"] == "medium"
        assert data["min_players"] == 3
        assert data["max_players"] == 4
        assert data["is_active"] is True
        assert data["archived_at"] is None
        assert sorted(data["tags"]) == ["strategy", "trading"]

        # Location header
        assert "location" in response.headers
        assert data["id"] in response.headers["location"]

    async def test_create_game_minimal_payload(self, api_client: AsyncClient) -> None:
        """Minimal valid payload (name only) returns 201."""
        response = await api_client.post(
            "/api/v1/games",
            json=_game_payload(),
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 201
        data = response.json()["data"]
        assert data["name"] == "Catan"
        assert data["tags"] == []
        assert data["complexity"] is None

    async def test_create_game_missing_name_returns_422(self, api_client: AsyncClient) -> None:
        """Missing required name field returns 422."""
        response = await api_client.post(
            "/api/v1/games",
            json={},
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 422


@pytest.mark.integration
class TestListGames:
    """GET /api/v1/games — list with filtering and pagination."""

    async def _create_game(self, client: AsyncClient, name: str, **kwargs: Any) -> dict[str, Any]:
        """Helper to create a game and return its data."""
        payload = _game_payload(name=name, **kwargs)
        resp = await client.post("/api/v1/games", json=payload)
        assert resp.status_code == 201
        return resp.json()["data"]

    async def test_list_games_returns_paginated_envelope(self, api_client: AsyncClient) -> None:
        """List returns ListEnvelope with pagination metadata."""
        await self._create_game(api_client, "Game A")
        await self._create_game(api_client, "Game B")

        response = await api_client.get(
            "/api/v1/games",
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 200
        body = response.json()
        assert "data" in body
        assert "meta" in body
        assert body["meta"]["request_id"] == _REQUEST_ID
        assert "pagination" in body["meta"]

        pagination = body["meta"]["pagination"]
        assert pagination["total_count"] >= 2
        assert pagination["page"] == 1

    async def test_list_games_search_filter(self, api_client: AsyncClient) -> None:
        """Search parameter filters by game name."""
        await self._create_game(api_client, "Unique Findable Name")
        await self._create_game(api_client, "Other Game")

        response = await api_client.get(
            "/api/v1/games",
            params={"search": "Un"},
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 200
        data = response.json()["data"]
        names = [g["name"] for g in data]
        assert "Unique Findable Name" in names

    async def test_list_games_tag_filter(self, api_client: AsyncClient) -> None:
        """Tags parameter filters games (AND logic)."""
        await self._create_game(api_client, "Tagged Game", tags=["strategy", "cooperative"])
        await self._create_game(api_client, "Untagged Game")

        response = await api_client.get(
            "/api/v1/games",
            params={"tags": "strategy"},
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 200
        data = response.json()["data"]
        names = [g["name"] for g in data]
        assert "Tagged Game" in names
        assert "Untagged Game" not in names

    async def test_list_games_pagination(self, api_client: AsyncClient) -> None:
        """Pagination parameters control page size and offset."""
        for i in range(5):
            await self._create_game(api_client, f"Paginated Game {i}")

        response = await api_client.get(
            "/api/v1/games",
            params={"page": 1, "page_size": 2},
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 200
        body = response.json()
        assert len(body["data"]) == 2
        assert body["meta"]["pagination"]["page_size"] == 2

    async def test_list_games_excludes_archived_by_default(self, api_client: AsyncClient) -> None:
        """Archived games are excluded when archived=false (default)."""
        game = await self._create_game(api_client, "To Be Archived")
        await api_client.post(f"/api/v1/games/{game['id']}/archive")

        response = await api_client.get(
            "/api/v1/games",
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 200
        names = [g["name"] for g in response.json()["data"]]
        assert "To Be Archived" not in names

    async def test_list_games_includes_archived_when_requested(
        self, api_client: AsyncClient
    ) -> None:
        """Archived games included when archived=true."""
        game = await self._create_game(api_client, "Archived But Visible")
        await api_client.post(f"/api/v1/games/{game['id']}/archive")

        response = await api_client.get(
            "/api/v1/games",
            params={"archived": "true"},
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 200
        names = [g["name"] for g in response.json()["data"]]
        assert "Archived But Visible" in names


@pytest.mark.integration
class TestListTags:
    """GET /api/v1/games/tags — list tags with counts."""

    async def test_list_tags_returns_tag_counts(self, api_client: AsyncClient) -> None:
        """Tags endpoint returns tags with their counts."""
        # Create games with overlapping tags
        await api_client.post(
            "/api/v1/games",
            json=_game_payload(name="Game 1", tags=["strategy", "family"]),
        )
        await api_client.post(
            "/api/v1/games",
            json=_game_payload(name="Game 2", tags=["strategy", "cooperative"]),
        )

        response = await api_client.get(
            "/api/v1/games/tags",
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 200
        body = response.json()
        assert "data" in body
        assert "meta" in body

        tags = {t["tag"]: t["count"] for t in body["data"]}
        assert tags["strategy"] == 2
        assert tags.get("family", 0) >= 1
        assert tags.get("cooperative", 0) >= 1

    async def test_list_tags_excludes_archived_game_tags(self, api_client: AsyncClient) -> None:
        """Tags from archived games are not counted."""
        resp = await api_client.post(
            "/api/v1/games",
            json=_game_payload(name="Archived Game", tags=["rare-tag"]),
        )
        game_id = resp.json()["data"]["id"]
        await api_client.post(f"/api/v1/games/{game_id}/archive")

        response = await api_client.get(
            "/api/v1/games/tags",
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 200
        tag_names = [t["tag"] for t in response.json()["data"]]
        assert "rare-tag" not in tag_names


@pytest.mark.integration
class TestGetGameDetail:
    """GET /api/v1/games/{game_id} — game detail with expansions."""

    async def test_get_game_returns_detail_envelope(self, api_client: AsyncClient) -> None:
        """Detail endpoint returns GameDetailResponse in envelope."""
        create_resp = await api_client.post(
            "/api/v1/games",
            json=_game_payload(
                name="Detail Game",
                tags=["rpg"],
                complexity="heavy",
            ),
        )
        game_id = create_resp.json()["data"]["id"]

        response = await api_client.get(
            f"/api/v1/games/{game_id}",
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["meta"]["request_id"] == _REQUEST_ID

        data = body["data"]
        assert data["id"] == game_id
        assert data["name"] == "Detail Game"
        assert data["complexity"] == "heavy"
        assert data["tags"] == ["rpg"]
        assert "document_count" in data
        assert "expansion_count" in data
        assert "expansions" in data
        assert isinstance(data["expansions"], list)

    async def test_get_game_not_found(self, api_client: AsyncClient) -> None:
        """Non-existent game returns 404."""
        fake_id = str(uuid.uuid4())
        response = await api_client.get(
            f"/api/v1/games/{fake_id}",
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 404


@pytest.mark.integration
class TestUpdateGame:
    """PATCH /api/v1/games/{game_id} — partial update."""

    async def test_update_game_partial(self, api_client: AsyncClient) -> None:
        """Partial update modifies only provided fields."""
        create_resp = await api_client.post(
            "/api/v1/games",
            json=_game_payload(name="Before Update", publisher="Original"),
        )
        game_id = create_resp.json()["data"]["id"]

        response = await api_client.patch(
            f"/api/v1/games/{game_id}",
            json={"publisher": "Updated Publisher"},
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["publisher"] == "Updated Publisher"
        assert data["name"] == "Before Update"

    async def test_update_game_tags_replacement(self, api_client: AsyncClient) -> None:
        """Tag update replaces the entire tag set."""
        create_resp = await api_client.post(
            "/api/v1/games",
            json=_game_payload(name="Tag Game", tags=["old-tag"]),
        )
        game_id = create_resp.json()["data"]["id"]

        response = await api_client.patch(
            f"/api/v1/games/{game_id}",
            json={"tags": ["new-tag-a", "new-tag-b"]},
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 200
        assert sorted(response.json()["data"]["tags"]) == ["new-tag-a", "new-tag-b"]

    async def test_update_game_not_found(self, api_client: AsyncClient) -> None:
        """Updating non-existent game returns 404."""
        fake_id = str(uuid.uuid4())
        response = await api_client.patch(
            f"/api/v1/games/{fake_id}",
            json={"name": "Nope"},
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 404


@pytest.mark.integration
class TestArchiveGame:
    """POST /api/v1/games/{game_id}/archive — soft delete."""

    async def test_archive_game_success(self, api_client: AsyncClient) -> None:
        """Archive sets archived_at and returns the game."""
        create_resp = await api_client.post(
            "/api/v1/games",
            json=_game_payload(name="Archivable"),
        )
        game_id = create_resp.json()["data"]["id"]

        response = await api_client.post(
            f"/api/v1/games/{game_id}/archive",
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["archived_at"] is not None
        assert data["is_active"] is False

    async def test_archive_already_archived_returns_409(self, api_client: AsyncClient) -> None:
        """Archiving an already-archived game returns 409."""
        create_resp = await api_client.post(
            "/api/v1/games",
            json=_game_payload(name="Double Archive"),
        )
        game_id = create_resp.json()["data"]["id"]

        await api_client.post(f"/api/v1/games/{game_id}/archive")
        response = await api_client.post(
            f"/api/v1/games/{game_id}/archive",
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 409


@pytest.mark.integration
class TestRestoreGame:
    """POST /api/v1/games/{game_id}/restore — un-archive."""

    async def test_restore_game_success(self, api_client: AsyncClient) -> None:
        """Restore clears archived_at and returns the game."""
        create_resp = await api_client.post(
            "/api/v1/games",
            json=_game_payload(name="Restorable"),
        )
        game_id = create_resp.json()["data"]["id"]

        await api_client.post(f"/api/v1/games/{game_id}/archive")
        response = await api_client.post(
            f"/api/v1/games/{game_id}/restore",
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["archived_at"] is None
        assert data["is_active"] is True

    async def test_restore_not_archived_returns_409(self, api_client: AsyncClient) -> None:
        """Restoring a non-archived game returns 409."""
        create_resp = await api_client.post(
            "/api/v1/games",
            json=_game_payload(name="Not Archived"),
        )
        game_id = create_resp.json()["data"]["id"]

        response = await api_client.post(
            f"/api/v1/games/{game_id}/restore",
            headers={"X-Request-ID": _REQUEST_ID},
        )

        assert response.status_code == 409


@pytest.mark.integration
class TestAuthEnforcement:
    """Verify authentication and role enforcement on game endpoints."""

    async def test_unauthenticated_request_returns_401(
        self, db_session: AsyncSession, _seed_user: User
    ) -> None:
        """Request without auth returns 401."""

        async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
            yield db_session

        app.dependency_overrides[get_db] = _override_get_db
        try:
            # bypass_auth=False (default) — no session cookie
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport,
                base_url="http://test",
            ) as client:
                response = await client.get(
                    "/api/v1/games",
                    headers={"X-Request-ID": _REQUEST_ID},
                )

            assert response.status_code == 401
        finally:
            app.dependency_overrides.clear()
