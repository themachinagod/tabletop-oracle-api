"""Session and SessionExpansion ORM models.

Maps to the ``sessions`` and ``session_expansions`` tables. These are
application game sessions (not auth sessions). The sessions table diverges
from Base: it uses ``last_active_at`` instead of ``updated_at`` and has
no ``updated_at`` column. SessionExpansion is a join table with a composite
primary key and no inherited columns.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import TIMESTAMP, Enum, ForeignKey, Integer, Text, func, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tabletop_oracle.models.base import MappedBase
from tabletop_oracle.models.enums import SessionStatus

if TYPE_CHECKING:
    from tabletop_oracle.models.expansion import Expansion
    from tabletop_oracle.models.game import Game
    from tabletop_oracle.models.message import Message
    from tabletop_oracle.models.user import User


class Session(MappedBase):
    """Game play session linking a user to a game.

    Uses ``last_active_at`` instead of ``updated_at``. Has ``created_at``
    but no ``updated_at`` column.

    Attributes:
        user_id: FK to the session owner.
        game_id: FK to the game being played.
        name: User-assigned session name.
        player_count: Number of players (optional).
        status: Session lifecycle status (active or archived).
        created_at: Session creation timestamp.
        last_active_at: Most recent activity timestamp.
    """

    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id"),
        nullable=False,
    )
    game_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("games.id"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    player_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[SessionStatus] = mapped_column(
        Enum(
            SessionStatus,
            native_enum=True,
            name="session_status",
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
        server_default="active",
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

    # Relationships
    user: Mapped[User] = relationship(back_populates="sessions")
    game: Mapped[Game] = relationship(back_populates="sessions")
    session_expansions: Mapped[list[SessionExpansion]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
    messages: Mapped[list[Message]] = relationship(back_populates="session")


class SessionExpansion(MappedBase):
    """Join table linking sessions to selected expansions.

    Composite PK (session_id, expansion_id). No inherited columns.

    Attributes:
        session_id: FK to the session (CASCADE on delete).
        expansion_id: FK to the selected expansion.
    """

    __tablename__ = "session_expansions"

    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"),
        primary_key=True,
    )
    expansion_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("expansions.id"),
        primary_key=True,
    )

    # Relationships
    session: Mapped[Session] = relationship(back_populates="session_expansions")
    expansion: Mapped[Expansion] = relationship(back_populates="session_expansions")
