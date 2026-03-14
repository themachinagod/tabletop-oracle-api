"""Authentication domain models.

Defines the ``CurrentUser`` dataclass injected into ``request.state``
by the session middleware. This is the in-memory representation of the
authenticated user available to all downstream route handlers and
dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID


@dataclass(frozen=True, slots=True)
class CurrentUser:
    """Authenticated user context injected by session middleware.

    Attributes:
        user_id: The user's primary key UUID.
        role: Application role — ``"player"`` or ``"curator"``.
        email: The user's email address.
        display_name: Human-readable display name.
        session_id: The auth session token ID (not game session).
    """

    user_id: UUID
    role: str
    email: str
    display_name: str
    session_id: str
