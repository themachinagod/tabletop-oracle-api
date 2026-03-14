"""Expansion ORM model.

Maps to the ``expansions`` table. Follows the standard Base pattern with
an additional ``archived_at`` column for soft-delete.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import TIMESTAMP, ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tabletop_oracle.models.base import Base

if TYPE_CHECKING:
    from tabletop_oracle.models.document import Document
    from tabletop_oracle.models.game import Game
    from tabletop_oracle.models.session import SessionExpansion


class Expansion(Base):
    """Game expansion entry, scoped to a parent game.

    Attributes:
        game_id: FK to the parent game (CASCADE on delete).
        name: Expansion title.
        year_published: Publication year (optional).
        description: Free-text description (optional).
        archived_at: Soft-delete timestamp; NULL means active.
    """

    __tablename__ = "expansions"

    game_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("games.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    year_published: Mapped[int | None] = mapped_column(Integer, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    archived_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=True,
    )

    # Relationships
    game: Mapped[Game] = relationship(back_populates="expansions")
    documents: Mapped[list[Document]] = relationship(back_populates="expansion")
    session_expansions: Mapped[list[SessionExpansion]] = relationship(
        back_populates="expansion",
    )
