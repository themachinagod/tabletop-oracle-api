"""First-login user service — find-or-create logic for OAuth users.

Handles three paths on login:
1. Known OAuth identity → update last_login_at, return user
2. Known email with no OAuth binding → bind OAuth identity (seed data)
3. New user → create with player role (or curator if bootstrap email)

Raises ConflictError if an email is already bound to a different OAuth
identity, and AuthenticationError if the account is disabled.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import select

from tabletop_oracle.auth.bootstrap import is_bootstrap_curator
from tabletop_oracle.errors.exceptions import AuthenticationError, ConflictError
from tabletop_oracle.models.enums import OAuthProvider, UserRole, UserStatus
from tabletop_oracle.models.user import User

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def find_or_create_user(
    db: AsyncSession,
    provider: OAuthProvider,
    subject_id: str,
    email: str,
    display_name: str,
) -> User:
    """Find an existing user or create a new one from OAuth claims.

    Lookup order:
    1. Match by (oauth_provider, oauth_subject_id) — returning user
    2. Match by email with oauth_subject_id IS NULL — seed data binding
    3. Match by email with different OAuth identity — ConflictError
    4. No match — create new user

    Args:
        db: Active async database session (caller owns transaction).
        provider: OAuth identity provider enum value.
        subject_id: Provider-specific subject identifier.
        email: User's email address from ID token claims.
        display_name: User's display name from ID token claims.

    Returns:
        The found or newly created User instance.

    Raises:
        AuthenticationError: If the user account is disabled.
        ConflictError: If the email is bound to a different OAuth identity.
    """
    # Path 1: lookup by OAuth identity
    stmt = select(User).where(
        User.oauth_provider == provider,
        User.oauth_subject_id == subject_id,
    )
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if user is not None:
        _check_disabled(user)
        user.last_login_at = datetime.now(UTC)
        await db.flush()
        return user

    # Path 2/3: lookup by email
    stmt_email = select(User).where(User.email == email)
    result_email = await db.execute(stmt_email)
    existing_by_email = result_email.scalar_one_or_none()

    if existing_by_email is not None:
        if existing_by_email.oauth_subject_id is None:
            # Seed data binding: attach OAuth identity to existing record
            _check_disabled(existing_by_email)
            existing_by_email.oauth_provider = provider
            existing_by_email.oauth_subject_id = subject_id
            existing_by_email.last_login_at = datetime.now(UTC)
            await db.flush()
            return existing_by_email

        # Email already bound to different OAuth identity
        raise ConflictError(f"Email '{email}' is already associated with a different account")

    # Path 4: create new user
    role = UserRole.CURATOR if is_bootstrap_curator(email) else UserRole.PLAYER
    now = datetime.now(UTC)
    new_user = User(
        oauth_provider=provider,
        oauth_subject_id=subject_id,
        email=email,
        display_name=display_name,
        role=role,
        status=UserStatus.ACTIVE,
        created_at=now,
        last_login_at=now,
    )
    db.add(new_user)
    await db.flush()
    return new_user


def _check_disabled(user: User) -> None:
    """Raise AuthenticationError if the user account is disabled.

    Args:
        user: The user to check.

    Raises:
        AuthenticationError: If user.status is DISABLED.
    """
    if user.status == UserStatus.DISABLED:
        raise AuthenticationError("Account is disabled")
