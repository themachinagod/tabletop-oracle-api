"""Message, Citation, and ContextAttachment ORM models.

Maps to the ``messages``, ``citations``, and ``context_attachments`` tables.
Messages have ``created_at`` but no ``updated_at``. Citations and
ContextAttachments have no timestamp columns at all.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import TIMESTAMP, Enum, Float, ForeignKey, Integer, Text, func, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tabletop_oracle.models.base import MappedBase
from tabletop_oracle.models.enums import (
    ContextAttachmentType,
    DocumentType,
    MessageType,
)

if TYPE_CHECKING:
    from tabletop_oracle.models.session import Session


class Message(MappedBase):
    """Chat message within a game session.

    Has ``created_at`` but no ``updated_at`` (messages are immutable after
    creation). Supports self-referential threading via ``in_reply_to_id``.

    Attributes:
        session_id: FK to the owning session.
        type: Message classification (user_question, ai_answer, etc.).
        content: Message text content.
        sequence: Monotonically increasing sequence within the session.
        in_reply_to_id: Self-referential FK for threading (optional).
        confidence_score: AI confidence in the answer, 0.0-1.0 (optional).
        processing_duration_ms: Time to generate the response in ms (optional).
        created_at: Message creation timestamp.
    """

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sessions.id"),
        nullable=False,
    )
    type: Mapped[MessageType] = mapped_column(
        Enum(
            MessageType,
            native_enum=True,
            name="message_type",
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    in_reply_to_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("messages.id"),
        nullable=True,
    )
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    processing_duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    session: Mapped[Session] = relationship(back_populates="messages")
    reply_to: Mapped[Message | None] = relationship(
        remote_side="Message.id",
        foreign_keys=[in_reply_to_id],
    )
    citations: Mapped[list[Citation]] = relationship(
        back_populates="message",
        cascade="all, delete-orphan",
    )
    context_attachments: Mapped[list[ContextAttachment]] = relationship(
        back_populates="message",
        cascade="all, delete-orphan",
    )


class Citation(MappedBase):
    """Source citation attached to an AI response message.

    No timestamp columns. ``document_name`` and ``document_type`` are
    intentionally denormalised to preserve the citation's historical accuracy
    even if the source document is later renamed or reclassified.

    Attributes:
        message_id: FK to the AI response message (CASCADE on delete).
        document_id: FK to the cited document.
        document_name: Denormalised document name at citation time.
        document_type: Denormalised document type at citation time.
        section_path: Document section path (optional).
        page_number: Page number reference (optional).
        excerpt: Quoted excerpt from the source.
    """

    __tablename__ = "citations"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    message_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id"),
        nullable=False,
    )
    document_name: Mapped[str] = mapped_column(Text, nullable=False)
    document_type: Mapped[DocumentType] = mapped_column(
        Enum(
            DocumentType,
            native_enum=True,
            name="document_type",
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
    )
    section_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    excerpt: Mapped[str] = mapped_column(Text, nullable=False)

    # Relationships
    message: Mapped[Message] = relationship(back_populates="citations")


class ContextAttachment(MappedBase):
    """User-provided context attached to a message.

    No timestamp columns. Content is either inline text or a file reference,
    enforced by a CHECK constraint at the database level.

    Attributes:
        message_id: FK to the owning message (CASCADE on delete).
        type: Attachment type (text or image).
        content: Inline text content (required when type is 'text').
        file_path: Blob store path (required when type is 'image').
        file_name: Original file name (optional).
        file_size: File size in bytes (optional).
        processed_description: AI-generated description of the attachment (optional).
    """

    __tablename__ = "context_attachments"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    message_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
    )
    type: Mapped[ContextAttachmentType] = mapped_column(
        Enum(
            ContextAttachmentType,
            native_enum=True,
            name="context_attachment_type",
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
    )
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    processed_description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    message: Mapped[Message] = relationship(back_populates="context_attachments")
