"""AuthSession ORM model.

Maps to the ``auth_sessions`` table. This is an authentication session
(not a game play session). The table stores server-side session state
for authenticated users, following ADR-003.

The ``id`` column is TEXT (an opaque base64url-encoded token), not UUID.
The table diverges from the standard Base pattern: no ``updated_at``
column, uses ``last_active_at`` and ``expires_at`` instead.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import TIMESTAMP, ForeignKey, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tabletop_oracle.models.base import MappedBase

if TYPE_CHECKING:
    from tabletop_oracle.models.user import User


class AuthSession(MappedBase):
    """Server-side authentication session linked to a user.

    Stores opaque session tokens with sliding expiry. Session IDs are
    256-bit cryptographically random values (base64url-encoded).

    Attributes:
        id: Opaque session token (TEXT, not UUID).
        user_id: FK to the authenticated user (CASCADE on delete).
        created_at: Session creation timestamp.
        last_active_at: Most recent request timestamp (sliding expiry).
        expires_at: Absolute session expiry timestamp.
        user_agent: Client User-Agent string (optional).
        ip_address: Client IP address (optional).
    """

    __tablename__ = "auth_sessions"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    last_active_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    expires_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
    )
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)
    ip_address: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    user: Mapped[User] = relationship(back_populates="auth_sessions")
