"""PlayerFeedback ORM model.

Maps to the ``player_feedback`` table. Has ``created_at`` but no
``updated_at`` (feedback is immutable after submission). ``session_id``
and ``game_id`` are intentionally denormalised for efficient aggregation.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import TIMESTAMP, Enum, ForeignKey, Text, func, text
from sqlalchemy.orm import Mapped, mapped_column

from tabletop_oracle.models.base import MappedBase
from tabletop_oracle.models.enums import FeedbackRating


class PlayerFeedback(MappedBase):
    """Player feedback on an AI response message.

    One feedback per message (enforced by unique constraint on message_id).
    ``session_id`` and ``game_id`` are denormalised from the message's
    session for efficient aggregation queries.

    Attributes:
        message_id: FK to the rated message (unique — one feedback per message).
        session_id: Denormalised FK to the session.
        game_id: Denormalised FK to the game.
        rating: Positive or negative rating.
        comment: Optional free-text comment.
        created_at: Feedback submission timestamp.
    """

    __tablename__ = "player_feedback"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    message_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("messages.id"),
        unique=True,
        nullable=False,
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sessions.id"),
        nullable=False,
    )
    game_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("games.id"),
        nullable=False,
    )
    rating: Mapped[FeedbackRating] = mapped_column(
        Enum(FeedbackRating, native_enum=True, name="feedback_rating"),
        nullable=False,
    )
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
