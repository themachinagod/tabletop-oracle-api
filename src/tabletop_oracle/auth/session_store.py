"""Server-side session management service.

Handles the full auth-session lifecycle: creation, lookup, sliding
expiry (throttled), single/bulk deletion, and expired-session cleanup.
Operates on the ``auth_sessions`` table via the :class:`AuthSession` model.

All public methods accept an ``AsyncSession`` — the caller owns the
database session and transaction boundaries.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import CursorResult, delete, select, update

from tabletop_oracle.config import settings
from tabletop_oracle.models.auth_session import AuthSession

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

# Touch throttle: only update last_active_at / expires_at if the
# previous activity is older than this threshold.
_TOUCH_THROTTLE = timedelta(hours=1)


class SessionStore:
    """CRUD service for authentication sessions.

    All methods are stateless class methods operating on the injected
    ``AsyncSession``.  The session timeout is read from application
    settings at call time so that tests can override it.
    """

    @staticmethod
    def _generate_session_id() -> str:
        """Return a 43-character URL-safe base64 session token.

        Uses :func:`secrets.token_urlsafe` with 32 bytes (256 bits) of
        cryptographic randomness.
        """
        return secrets.token_urlsafe(32)

    @staticmethod
    def _session_timeout() -> timedelta:
        """Return the configured session timeout as a timedelta."""
        return timedelta(days=settings.auth_session_timeout_days)

    @staticmethod
    async def create(
        db: AsyncSession,
        user_id: UUID,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> str:
        """Create a new auth session and return the session ID.

        Args:
            db: Active async database session.
            user_id: UUID of the authenticated user.
            user_agent: Client User-Agent header value.
            ip_address: Client IP address.

        Returns:
            The generated session ID (43-char URL-safe base64 string).
        """
        session_id = SessionStore._generate_session_id()
        now = datetime.now(UTC)
        expires_at = now + SessionStore._session_timeout()

        auth_session = AuthSession(
            id=session_id,
            user_id=user_id,
            created_at=now,
            last_active_at=now,
            expires_at=expires_at,
            user_agent=user_agent,
            ip_address=ip_address,
        )
        db.add(auth_session)
        await db.flush()
        return session_id

    @staticmethod
    async def get(db: AsyncSession, session_id: str) -> AuthSession | None:
        """Fetch a session by ID.

        Args:
            db: Active async database session.
            session_id: The session token to look up.

        Returns:
            The :class:`AuthSession` instance, or ``None`` if not found.
        """
        stmt = select(AuthSession).where(AuthSession.id == session_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def touch(db: AsyncSession, session_id: str) -> None:
        """Extend the session's sliding expiry window.

        Updates ``last_active_at`` and ``expires_at`` only when the
        previous ``last_active_at`` is older than the throttle threshold
        (1 hour). This avoids a DB write on every single request.

        Args:
            db: Active async database session.
            session_id: The session token to refresh.
        """
        now = datetime.now(UTC)
        throttle_cutoff = now - _TOUCH_THROTTLE

        stmt = (
            update(AuthSession)
            .where(
                AuthSession.id == session_id,
                AuthSession.last_active_at < throttle_cutoff,
            )
            .values(
                last_active_at=now,
                expires_at=now + SessionStore._session_timeout(),
            )
        )
        await db.execute(stmt)

    @staticmethod
    async def delete(db: AsyncSession, session_id: str) -> None:
        """Delete a single session (logout).

        Args:
            db: Active async database session.
            session_id: The session token to remove.
        """
        stmt = delete(AuthSession).where(AuthSession.id == session_id)
        await db.execute(stmt)

    @staticmethod
    async def delete_all_for_user(db: AsyncSession, user_id: UUID) -> int:
        """Delete every session belonging to a user.

        Args:
            db: Active async database session.
            user_id: UUID of the user whose sessions should be removed.

        Returns:
            Number of sessions deleted.
        """
        stmt = delete(AuthSession).where(AuthSession.user_id == user_id)
        result: CursorResult[tuple[()]] = await db.execute(stmt)  # type: ignore[assignment]
        return result.rowcount

    @staticmethod
    async def cleanup_expired(db: AsyncSession) -> int:
        """Delete all sessions whose ``expires_at`` is in the past.

        Args:
            db: Active async database session.

        Returns:
            Number of expired sessions deleted.
        """
        now = datetime.now(UTC)
        stmt = delete(AuthSession).where(AuthSession.expires_at < now)
        result: CursorResult[tuple[()]] = await db.execute(stmt)  # type: ignore[assignment]
        return result.rowcount
