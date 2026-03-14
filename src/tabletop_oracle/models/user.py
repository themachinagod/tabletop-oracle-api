"""User ORM model.

Maps to the ``users`` table. The users table diverges from the standard
Base pattern: it has no ``updated_at`` column (user profile changes are
rare in V1) and uses ``last_login_at`` instead.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import TIMESTAMP, Enum, Text, func, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tabletop_oracle.models.base import MappedBase
from tabletop_oracle.models.enums import OAuthProvider, UserRole, UserStatus

if TYPE_CHECKING:
    from tabletop_oracle.models.auth_session import AuthSession
    from tabletop_oracle.models.document import DocumentVersion
    from tabletop_oracle.models.game import Game
    from tabletop_oracle.models.session import Session


class User(MappedBase):
    """User account linked to an OAuth identity provider.

    Attributes:
        id: UUID primary key.
        oauth_provider: Identity provider used for authentication.
        oauth_subject_id: Provider-specific subject identifier (nullable for
            future local accounts).
        email: Unique email address.
        display_name: User-facing display name.
        role: Application role (player or curator).
        status: Account status (active or disabled).
        created_at: Account creation timestamp.
        last_login_at: Most recent authentication timestamp.
    """

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    oauth_provider: Mapped[OAuthProvider] = mapped_column(
        Enum(OAuthProvider, native_enum=True, name="oauth_provider"),
        nullable=False,
    )
    oauth_subject_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    email: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, native_enum=True, name="user_role"),
        nullable=False,
        server_default="player",
    )
    status: Mapped[UserStatus] = mapped_column(
        Enum(UserStatus, native_enum=True, name="user_status"),
        nullable=False,
        server_default="active",
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    last_login_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    games: Mapped[list[Game]] = relationship(back_populates="creator")
    sessions: Mapped[list[Session]] = relationship(back_populates="user")
    auth_sessions: Mapped[list[AuthSession]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
    document_versions: Mapped[list[DocumentVersion]] = relationship(
        back_populates="uploader",
    )
