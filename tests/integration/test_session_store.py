"""Integration tests for SessionStore against real PostgreSQL.

Validates full CRUD lifecycle including create, get, touch (with
throttle), single/bulk delete, and expired-session cleanup.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text, update

from tabletop_oracle.auth.session_store import SessionStore
from tabletop_oracle.models.auth_session import AuthSession

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

pytestmark = pytest.mark.integration


async def _create_test_user(db: AsyncSession) -> uuid.UUID:
    """Insert a minimal user via raw SQL and return the user ID.

    Uses raw SQL to avoid coupling to the User ORM model's enum
    mapping behaviour, which differs between SQLAlchemy versions.
    """
    user_id = uuid.uuid4()
    email = f"test-{uuid.uuid4().hex[:8]}@example.com"
    await db.execute(
        text(
            "INSERT INTO users (id, oauth_provider, oauth_subject_id, email, display_name) "
            "VALUES (:id, :provider, :sub, :email, :name)"
        ),
        {
            "id": str(user_id),
            "provider": "google",
            "sub": "test-subject-id",
            "email": email,
            "name": "Test User",
        },
    )
    return user_id


class TestSessionStoreCreate:
    """SessionStore.create() inserts a valid session row."""

    async def test_create_returns_session_id(self, db_session: AsyncSession) -> None:
        """create() returns a 43-character session ID."""
        user_id = await _create_test_user(db_session)
        sid = await SessionStore.create(db_session, user_id)
        assert isinstance(sid, str)
        assert len(sid) == 43

    async def test_create_stores_session_in_db(self, db_session: AsyncSession) -> None:
        """Created session is retrievable from the database."""
        user_id = await _create_test_user(db_session)
        sid = await SessionStore.create(db_session, user_id, user_agent="TestAgent/1.0")
        session = await SessionStore.get(db_session, sid)
        assert session is not None
        assert session.id == sid
        assert session.user_id == user_id
        assert session.user_agent == "TestAgent/1.0"

    async def test_create_sets_correct_expires_at(self, db_session: AsyncSession) -> None:
        """expires_at is approximately now + configured timeout."""
        user_id = await _create_test_user(db_session)
        before = datetime.now(UTC)
        sid = await SessionStore.create(db_session, user_id)
        after = datetime.now(UTC)

        session = await SessionStore.get(db_session, sid)
        assert session is not None

        timeout = SessionStore._session_timeout()
        assert before + timeout <= session.expires_at <= after + timeout

    async def test_create_stores_ip_address(self, db_session: AsyncSession) -> None:
        """ip_address is persisted when provided."""
        user_id = await _create_test_user(db_session)
        sid = await SessionStore.create(db_session, user_id, ip_address="192.168.1.1")
        session = await SessionStore.get(db_session, sid)
        assert session is not None
        assert session.ip_address == "192.168.1.1"

    async def test_create_optional_fields_default_none(self, db_session: AsyncSession) -> None:
        """user_agent and ip_address default to None."""
        user_id = await _create_test_user(db_session)
        sid = await SessionStore.create(db_session, user_id)
        session = await SessionStore.get(db_session, sid)
        assert session is not None
        assert session.user_agent is None
        assert session.ip_address is None


class TestSessionStoreGet:
    """SessionStore.get() retrieves or returns None."""

    async def test_get_returns_none_for_missing(self, db_session: AsyncSession) -> None:
        """get() returns None for a nonexistent session ID."""
        result = await SessionStore.get(db_session, "nonexistent-session-id")
        assert result is None

    async def test_get_returns_session(self, db_session: AsyncSession) -> None:
        """get() returns the AuthSession model for a valid ID."""
        user_id = await _create_test_user(db_session)
        sid = await SessionStore.create(db_session, user_id)
        session = await SessionStore.get(db_session, sid)
        assert session is not None
        assert isinstance(session, AuthSession)
        assert session.id == sid


class TestSessionStoreTouch:
    """SessionStore.touch() implements throttled sliding expiry."""

    async def test_touch_extends_expiry_when_stale(self, db_session: AsyncSession) -> None:
        """touch() updates expires_at when last_active_at is > 1 hour old."""
        user_id = await _create_test_user(db_session)
        sid = await SessionStore.create(db_session, user_id)

        # Manually set last_active_at to 2 hours ago
        two_hours_ago = datetime.now(UTC) - timedelta(hours=2)
        stmt = (
            update(AuthSession).where(AuthSession.id == sid).values(last_active_at=two_hours_ago)
        )
        await db_session.execute(stmt)

        before_touch = datetime.now(UTC)
        await SessionStore.touch(db_session, sid)

        session = await SessionStore.get(db_session, sid)
        assert session is not None
        # last_active_at should be updated to approximately now
        assert session.last_active_at >= before_touch
        # expires_at should be extended
        timeout = SessionStore._session_timeout()
        assert session.expires_at >= before_touch + timeout

    async def test_touch_throttled_when_recent(self, db_session: AsyncSession) -> None:
        """touch() does NOT update when last_active_at is < 1 hour old."""
        user_id = await _create_test_user(db_session)
        sid = await SessionStore.create(db_session, user_id)

        # Get the original timestamps
        session_before = await SessionStore.get(db_session, sid)
        assert session_before is not None
        original_last_active = session_before.last_active_at
        original_expires = session_before.expires_at

        # Touch immediately (last_active_at is now, well within throttle)
        await SessionStore.touch(db_session, sid)

        # Refresh to see current state
        await db_session.refresh(session_before)
        assert session_before.last_active_at == original_last_active
        assert session_before.expires_at == original_expires


class TestSessionStoreDelete:
    """SessionStore.delete() removes a single session."""

    async def test_delete_removes_session(self, db_session: AsyncSession) -> None:
        """delete() removes the session from the database."""
        user_id = await _create_test_user(db_session)
        sid = await SessionStore.create(db_session, user_id)
        await SessionStore.delete(db_session, sid)
        result = await SessionStore.get(db_session, sid)
        assert result is None

    async def test_delete_nonexistent_is_noop(self, db_session: AsyncSession) -> None:
        """delete() does not raise for a missing session ID."""
        await SessionStore.delete(db_session, "does-not-exist")


class TestSessionStoreDeleteAllForUser:
    """SessionStore.delete_all_for_user() removes all user sessions."""

    async def test_deletes_all_user_sessions(self, db_session: AsyncSession) -> None:
        """All sessions for the specified user are deleted."""
        user_id = await _create_test_user(db_session)
        sid1 = await SessionStore.create(db_session, user_id)
        sid2 = await SessionStore.create(db_session, user_id)
        sid3 = await SessionStore.create(db_session, user_id)

        count = await SessionStore.delete_all_for_user(db_session, user_id)
        assert count == 3

        for sid in (sid1, sid2, sid3):
            assert await SessionStore.get(db_session, sid) is None

    async def test_does_not_delete_other_users_sessions(self, db_session: AsyncSession) -> None:
        """Sessions belonging to other users are not affected."""
        user_a = await _create_test_user(db_session)
        user_b = await _create_test_user(db_session)
        await SessionStore.create(db_session, user_a)
        sid_b = await SessionStore.create(db_session, user_b)

        await SessionStore.delete_all_for_user(db_session, user_a)

        assert await SessionStore.get(db_session, sid_b) is not None

    async def test_returns_zero_when_no_sessions(self, db_session: AsyncSession) -> None:
        """Returns 0 when the user has no sessions."""
        user_id = await _create_test_user(db_session)
        count = await SessionStore.delete_all_for_user(db_session, user_id)
        assert count == 0


class TestSessionStoreCleanupExpired:
    """SessionStore.cleanup_expired() deletes past-due sessions."""

    async def test_cleanup_deletes_expired(self, db_session: AsyncSession) -> None:
        """Sessions with expires_at in the past are removed."""
        user_id = await _create_test_user(db_session)
        sid = await SessionStore.create(db_session, user_id)

        # Manually expire the session
        past = datetime.now(UTC) - timedelta(hours=1)
        stmt = update(AuthSession).where(AuthSession.id == sid).values(expires_at=past)
        await db_session.execute(stmt)

        count = await SessionStore.cleanup_expired(db_session)
        assert count == 1
        assert await SessionStore.get(db_session, sid) is None

    async def test_cleanup_preserves_valid_sessions(self, db_session: AsyncSession) -> None:
        """Sessions with future expires_at are not deleted."""
        user_id = await _create_test_user(db_session)
        sid = await SessionStore.create(db_session, user_id)

        count = await SessionStore.cleanup_expired(db_session)
        assert count == 0
        assert await SessionStore.get(db_session, sid) is not None

    async def test_cleanup_mixed_expired_and_valid(self, db_session: AsyncSession) -> None:
        """Only expired sessions are removed; valid ones remain."""
        user_id = await _create_test_user(db_session)
        expired_sid = await SessionStore.create(db_session, user_id)
        valid_sid = await SessionStore.create(db_session, user_id)

        # Expire one session
        past = datetime.now(UTC) - timedelta(hours=1)
        stmt = update(AuthSession).where(AuthSession.id == expired_sid).values(expires_at=past)
        await db_session.execute(stmt)

        count = await SessionStore.cleanup_expired(db_session)
        assert count == 1
        assert await SessionStore.get(db_session, expired_sid) is None
        assert await SessionStore.get(db_session, valid_sid) is not None
