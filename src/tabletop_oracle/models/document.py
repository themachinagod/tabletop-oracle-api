"""Document, DocumentVersion, and DocumentChunk ORM models.

Maps to the ``documents``, ``document_versions``, and ``document_chunks``
tables. Documents diverge from the standard Base pattern: they use
``uploaded_at`` instead of ``created_at`` and have no ``created_at``
column. DocumentVersions have neither ``created_at`` nor ``updated_at``.
DocumentChunks have ``created_at`` only (append-only).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    TIMESTAMP,
    Boolean,
    Enum,
    ForeignKey,
    Integer,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
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
        archived_at: Soft-delete timestamp (null = active).
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
    archived_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=True,
    )

    # Relationships
    game: Mapped[Game] = relationship(back_populates="documents")
    expansion: Mapped[Expansion | None] = relationship(back_populates="documents")
    versions: Mapped[list[DocumentVersion]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )
    chunks: Mapped[list[DocumentChunk]] = relationship(
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
    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="version",
        cascade="all, delete-orphan",
    )


class DocumentChunk(MappedBase):
    """Content chunk produced by the ingestion pipeline.

    Chunks are the pipeline's deliverable: structured, traceable pieces
    of document content consumed by the KG engine and used for preview.
    Append-only -- has ``created_at`` but no ``updated_at``.

    Attributes:
        document_id: FK to the parent document (CASCADE on delete).
        version_id: FK to the specific document version (CASCADE on delete).
        chunk_index: Sequential position within the document version.
        chunk_type: Content type (text, table, image_description).
        content: The chunk text content.
        section_path: Hierarchical section context (e.g. "Combat > Ranged Attacks").
        heading: Nearest heading this chunk falls under.
        page_number: Source page number (PDF only, null for other formats).
        token_estimate: Approximate token count for LLM context budgeting.
        chunk_metadata: JSONB for format-specific metadata (maps to DB column ``metadata``).
        created_at: When the chunk was created.
    """

    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "version_id", "chunk_index", name="uq_chunk_index"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document_versions.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_type: Mapped[str] = mapped_column(Text, nullable=False, server_default="text")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    section_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    heading: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    token_estimate: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB(astext_type=Text()),
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    document: Mapped[Document] = relationship(back_populates="chunks")
    version: Mapped[DocumentVersion] = relationship(back_populates="chunks")
