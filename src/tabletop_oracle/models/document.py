"""Document and DocumentVersion ORM models.

Maps to the ``documents`` and ``document_versions`` tables. Documents
diverge from the standard Base pattern: they use ``uploaded_at`` instead
of ``created_at`` and have no ``created_at`` column. DocumentVersions
have neither ``created_at`` nor ``updated_at``.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import TIMESTAMP, Boolean, Enum, ForeignKey, Integer, Text, func, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tabletop_oracle.models.base import MappedBase
from tabletop_oracle.models.enums import DocumentFormat, DocumentStatus, DocumentType

if TYPE_CHECKING:
    from tabletop_oracle.models.expansion import Expansion
    from tabletop_oracle.models.game import Game
    from tabletop_oracle.models.user import User


class Document(MappedBase):
    """Uploaded document metadata, scoped to a game.

    Uses ``uploaded_at`` instead of ``created_at``. Has ``updated_at``
    (trigger-managed) but no ``created_at``.

    Attributes:
        game_id: FK to the owning game.
        expansion_id: FK to the associated expansion (optional).
        name: Document display name.
        type: Content classification (core_rules, faq, errata, etc.).
        format: File format (pdf, markdown, text, html, docx).
        status: Processing status (uploaded, parsing, processed, error).
        current_version: Active version number (default 1).
        file_path: Blob store path to the current file.
        file_size: File size in bytes.
        error_message: Processing error details (optional).
        uploaded_at: When the document was first uploaded.
        processed_at: When processing completed (optional).
        updated_at: Last modification timestamp (trigger-managed).
    """

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    game_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("games.id"),
        nullable=False,
    )
    expansion_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("expansions.id"),
        nullable=True,
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    type: Mapped[DocumentType] = mapped_column(
        Enum(DocumentType, native_enum=True, name="document_type"),
        nullable=False,
    )
    format: Mapped[DocumentFormat] = mapped_column(
        Enum(DocumentFormat, native_enum=True, name="document_format"),
        nullable=False,
    )
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, native_enum=True, name="document_status"),
        nullable=False,
        server_default="uploaded",
    )
    current_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default="1",
    )
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    processed_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    game: Mapped[Game] = relationship(back_populates="documents")
    expansion: Mapped[Expansion | None] = relationship(back_populates="documents")
    versions: Mapped[list[DocumentVersion]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )


class DocumentVersion(MappedBase):
    """Versioned snapshot of a document file.

    Has neither ``created_at`` nor ``updated_at``. Uses ``uploaded_at``
    and ``processed_at`` as temporal markers.

    Attributes:
        document_id: FK to the parent document (CASCADE on delete).
        version_number: Sequential version number within the document.
        file_path: Blob store path to this version's file.
        file_size: File size in bytes.
        is_active: Whether this is the active version.
        uploaded_at: When this version was uploaded.
        processed_at: When processing completed (optional).
        uploaded_by: FK to the user who uploaded this version.
    """

    __tablename__ = "document_versions"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("TRUE"),
    )
    uploaded_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    processed_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=True,
    )
    uploaded_by: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id"),
        nullable=False,
    )

    # Relationships
    document: Mapped[Document] = relationship(back_populates="versions")
    uploader: Mapped[User] = relationship(back_populates="document_versions")
