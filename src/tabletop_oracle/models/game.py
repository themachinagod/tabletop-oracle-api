"""Game and GameTag ORM models.

Maps to the ``games`` and ``game_tags`` tables. Games follow the standard
Base pattern (id + created_at + updated_at). GameTag is a join table with
a composite primary key and no inherited columns.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import TIMESTAMP, Enum, ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tabletop_oracle.models.base import Base, MappedBase
from tabletop_oracle.models.enums import GameComplexity

if TYPE_CHECKING:
    from tabletop_oracle.models.document import Document
    from tabletop_oracle.models.expansion import Expansion
    from tabletop_oracle.models.session import Session
    from tabletop_oracle.models.user import User


class Game(Base):
    """Board game catalogue entry.

    Attributes:
        created_by: FK to the curator who created this game entry.
        name: Game title (indexed for full-text search).
        publisher: Publisher name (optional).
        year_published: Publication year (optional).
        edition: Edition descriptor (optional).
        min_players: Minimum player count (optional, paired with max_players).
        max_players: Maximum player count (optional, paired with min_players).
        description: Free-text description (optional).
        cover_image_url: Blob store path to cover image (optional).
        complexity: Complexity classification (optional).
        archived_at: Soft-delete timestamp; NULL means active.
    """

    __tablename__ = "games"

    created_by: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    publisher: Mapped[str | None] = mapped_column(Text, nullable=True)
    year_published: Mapped[int | None] = mapped_column(Integer, nullable=True)
    edition: Mapped[str | None] = mapped_column(Text, nullable=True)
    min_players: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_players: Mapped[int | None] = mapped_column(Integer, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    cover_image_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    complexity: Mapped[GameComplexity | None] = mapped_column(
        Enum(
            GameComplexity,
            native_enum=True,
            name="game_complexity",
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=True,
    )
    archived_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=True,
    )

    # Relationships
    creator: Mapped[User] = relationship(back_populates="games")
    expansions: Mapped[list[Expansion]] = relationship(
        back_populates="game",
        cascade="all, delete-orphan",
    )
    documents: Mapped[list[Document]] = relationship(back_populates="game")
    tags: Mapped[list[GameTag]] = relationship(
        back_populates="game",
        cascade="all, delete-orphan",
    )
    sessions: Mapped[list[Session]] = relationship(back_populates="game")


class GameTag(MappedBase):
    """Tag associated with a game. Composite PK (game_id, tag).

    Attributes:
        game_id: FK to the owning game.
        tag: Tag string value.
    """

    __tablename__ = "game_tags"

    game_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("games.id", ondelete="CASCADE"),
        primary_key=True,
    )
    tag: Mapped[str] = mapped_column(Text, primary_key=True)

    # Relationships
    game: Mapped[Game] = relationship(back_populates="tags")
