"""ModelSlot and TokenUsageLog ORM models.

Maps to the ``model_slots`` and ``token_usage_log`` tables. ModelSlots
have ``updated_at`` but no ``created_at``. TokenUsageLog is append-only
with ``created_at`` but no ``updated_at``.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import TIMESTAMP, Enum, Float, ForeignKey, Integer, Text, func, text
from sqlalchemy.orm import Mapped, mapped_column

from tabletop_oracle.models.base import MappedBase
from tabletop_oracle.models.enums import ModelCapability


class ModelSlot(MappedBase):
    """AI model configuration slot, one per capability.

    Has ``updated_at`` (trigger-managed) but no ``created_at``.
    Uniqueness on ``capability`` ensures exactly one model per capability.

    Attributes:
        capability: The AI capability this slot serves (unique).
        provider: Model provider identifier (e.g., 'openai', 'anthropic').
        model_id: Provider-specific model identifier.
        max_tokens_per_call: Maximum tokens per API call (optional).
        temperature: Sampling temperature (optional).
        fallback_provider: Fallback provider if primary fails (optional).
        fallback_model_id: Fallback model identifier (optional).
        updated_at: Last modification timestamp (trigger-managed).
    """

    __tablename__ = "model_slots"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    capability: Mapped[ModelCapability] = mapped_column(
        Enum(
            ModelCapability,
            native_enum=True,
            name="model_capability",
            values_callable=lambda e: [m.value for m in e],
        ),
        unique=True,
        nullable=False,
    )
    provider: Mapped[str] = mapped_column(Text, nullable=False)
    model_id: Mapped[str] = mapped_column(Text, nullable=False)
    max_tokens_per_call: Mapped[int | None] = mapped_column(Integer, nullable=True)
    temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    fallback_provider: Mapped[str | None] = mapped_column(Text, nullable=True)
    fallback_model_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class TokenUsageLog(MappedBase):
    """Append-only log of AI model token consumption.

    Has ``created_at`` but no ``updated_at`` (immutable log entries).
    Attribution links are nullable because calls can originate from
    different contexts (user queries via session/message, document
    ingestion via document_id).

    Attributes:
        capability: The AI capability that consumed tokens.
        provider: Model provider used.
        model_id: Specific model used.
        input_tokens: Number of input tokens consumed.
        output_tokens: Number of output tokens generated.
        user_id: FK to the user who triggered the call (optional).
        session_id: FK to the session context (optional).
        message_id: FK to the message context (optional).
        document_id: FK to the document context (optional).
        created_at: When the token consumption occurred.
    """

    __tablename__ = "token_usage_log"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    capability: Mapped[ModelCapability] = mapped_column(
        Enum(
            ModelCapability,
            native_enum=True,
            name="model_capability",
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
    )
    provider: Mapped[str] = mapped_column(Text, nullable=False)
    model_id: Mapped[str] = mapped_column(Text, nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("sessions.id"),
        nullable=True,
    )
    message_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("messages.id"),
        nullable=True,
    )
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("documents.id"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
