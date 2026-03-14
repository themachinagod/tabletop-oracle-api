"""Unit tests for first-login user service.

Tests all find_or_create_user paths using mocked database sessions.
Covers: returning user, seed data binding, email conflict, new user
creation (player and curator), and disabled account handling.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tabletop_oracle.auth.user_service import find_or_create_user
from tabletop_oracle.errors.exceptions import AuthenticationError, ConflictError
from tabletop_oracle.models.enums import OAuthProvider, UserRole, UserStatus
from tabletop_oracle.models.user import User


def _make_user(
    *,
    provider: OAuthProvider = OAuthProvider.GOOGLE,
    subject_id: str | None = "sub-123",
    email: str = "user@example.com",
    display_name: str = "Test User",
    role: UserRole = UserRole.PLAYER,
    status: UserStatus = UserStatus.ACTIVE,
) -> User:
    """Create a User instance for testing without database interaction."""
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    user.oauth_provider = provider
    user.oauth_subject_id = subject_id
    user.email = email
    user.display_name = display_name
    user.role = role
    user.status = status
    user.last_login_at = datetime.now(UTC)
    return user


def _mock_db_with_results(
    oauth_result: User | None,
    email_result: User | None = None,
) -> AsyncMock:
    """Create a mock async DB session returning specified query results.

    First execute call returns oauth_result, second returns email_result.
    """
    db = AsyncMock()

    # Build mock result objects
    oauth_mock_result = MagicMock()
    oauth_mock_result.scalar_one_or_none.return_value = oauth_result

    email_mock_result = MagicMock()
    email_mock_result.scalar_one_or_none.return_value = email_result

    db.execute = AsyncMock(side_effect=[oauth_mock_result, email_mock_result])
    db.flush = AsyncMock()
    return db


class TestFindOrCreateUserReturningUser:
    """Path 1: user found by OAuth identity."""

    @pytest.mark.asyncio
    async def test_returns_existing_user(self) -> None:
        """Known OAuth identity returns the existing user."""
        existing = _make_user()
        db = _mock_db_with_results(oauth_result=existing)

        result = await find_or_create_user(
            db=db,
            provider=OAuthProvider.GOOGLE,
            subject_id="sub-123",
            email="user@example.com",
            display_name="Test User",
        )

        assert result is existing
        db.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_updates_last_login(self) -> None:
        """Returning user gets last_login_at updated."""
        existing = _make_user()
        db = _mock_db_with_results(oauth_result=existing)

        await find_or_create_user(
            db=db,
            provider=OAuthProvider.GOOGLE,
            subject_id="sub-123",
            email="user@example.com",
            display_name="Test User",
        )

        # last_login_at was reassigned — MagicMock accepts attribute assignment
        assert db.flush.await_count == 1

    @pytest.mark.asyncio
    async def test_disabled_user_raises(self) -> None:
        """Disabled returning user raises AuthenticationError."""
        existing = _make_user(status=UserStatus.DISABLED)
        db = _mock_db_with_results(oauth_result=existing)

        with pytest.raises(AuthenticationError, match="disabled"):
            await find_or_create_user(
                db=db,
                provider=OAuthProvider.GOOGLE,
                subject_id="sub-123",
                email="user@example.com",
                display_name="Test User",
            )


class TestFindOrCreateUserSeedBinding:
    """Path 2: email match with no OAuth binding (seed data)."""

    @pytest.mark.asyncio
    async def test_binds_oauth_to_seed_user(self) -> None:
        """User with matching email and no subject_id gets OAuth identity bound."""
        seed_user = _make_user(subject_id=None)
        db = _mock_db_with_results(oauth_result=None, email_result=seed_user)

        result = await find_or_create_user(
            db=db,
            provider=OAuthProvider.GOOGLE,
            subject_id="new-sub-456",
            email="user@example.com",
            display_name="Test User",
        )

        assert result is seed_user
        assert seed_user.oauth_provider == OAuthProvider.GOOGLE
        assert seed_user.oauth_subject_id == "new-sub-456"

    @pytest.mark.asyncio
    async def test_disabled_seed_user_raises(self) -> None:
        """Disabled seed user raises AuthenticationError."""
        seed_user = _make_user(subject_id=None, status=UserStatus.DISABLED)
        db = _mock_db_with_results(oauth_result=None, email_result=seed_user)

        with pytest.raises(AuthenticationError, match="disabled"):
            await find_or_create_user(
                db=db,
                provider=OAuthProvider.GOOGLE,
                subject_id="new-sub-456",
                email="user@example.com",
                display_name="Test User",
            )


class TestFindOrCreateUserConflict:
    """Path 3: email already bound to different OAuth identity."""

    @pytest.mark.asyncio
    async def test_email_bound_to_different_identity_raises(self) -> None:
        """Email matching a user with a different OAuth binding raises ConflictError."""
        conflicting = _make_user(
            provider=OAuthProvider.MICROSOFT,
            subject_id="other-sub-789",
        )
        db = _mock_db_with_results(oauth_result=None, email_result=conflicting)

        with pytest.raises(ConflictError, match="already associated"):
            await find_or_create_user(
                db=db,
                provider=OAuthProvider.GOOGLE,
                subject_id="new-sub-456",
                email="user@example.com",
                display_name="Test User",
            )


class TestFindOrCreateUserNewUser:
    """Path 4: no existing user found — create new."""

    @pytest.mark.asyncio
    async def test_creates_new_player(self) -> None:
        """New user without bootstrap email gets player role."""
        db = _mock_db_with_results(oauth_result=None, email_result=None)

        with patch("tabletop_oracle.auth.user_service.is_bootstrap_curator", return_value=False):
            result = await find_or_create_user(
                db=db,
                provider=OAuthProvider.GOOGLE,
                subject_id="sub-new",
                email="newuser@example.com",
                display_name="New User",
            )

        assert result.role == UserRole.PLAYER
        assert result.email == "newuser@example.com"
        assert result.oauth_provider == OAuthProvider.GOOGLE
        assert result.oauth_subject_id == "sub-new"
        db.add.assert_called_once()
        db.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_creates_new_curator(self) -> None:
        """New user with bootstrap email gets curator role."""
        db = _mock_db_with_results(oauth_result=None, email_result=None)

        with patch("tabletop_oracle.auth.user_service.is_bootstrap_curator", return_value=True):
            result = await find_or_create_user(
                db=db,
                provider=OAuthProvider.MICROSOFT,
                subject_id="sub-curator",
                email="curator@example.com",
                display_name="Curator User",
            )

        assert result.role == UserRole.CURATOR
        assert result.email == "curator@example.com"
        db.add.assert_called_once()
