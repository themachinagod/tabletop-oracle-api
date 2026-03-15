"""Knowledge Graph ORM models.

Maps to the ``kg_concepts``, ``kg_concept_sources``, ``kg_associations``,
``kg_association_sources``, ``concept_embeddings``, and ``kg_audit_log``
tables. These tables store the derived semantic index built from document
chunks by the KG engine (EPIC-003).

Design reference: docs/architecture/epic-003-knowledge-graph-engine/design.md
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    TIMESTAMP,
    Boolean,
    ForeignKey,
    Integer,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tabletop_oracle.models.base import Base, MappedBase

if TYPE_CHECKING:
    from tabletop_oracle.models.document import Document, DocumentChunk, DocumentVersion
    from tabletop_oracle.models.expansion import Expansion
    from tabletop_oracle.models.game import Game
    from tabletop_oracle.models.session import Session


class KGConcept(Base):
    """Concept extracted from document chunks, scoped to a game.

    Follows the standard Base pattern (id + created_at + updated_at).
    Concepts are the nodes of the knowledge graph.

    Attributes:
        game_id: FK to the owning game. Enforces per-game KG isolation.
        name: Canonical concept name. Unique per game.
        semantic_type: Type discovered from content (e.g., "game_mechanic").
            Not from a fixed enum.
        description: Brief description derived from source text.
        canonical_id: Self-referencing FK for synonym resolution. Aliases
            point to their canonical concept. NULL for canonical concepts.
        session_id: NULL for V1 global KG. Extension point for session-level
            knowledge (PRD-005 Section 9).
    """

    __tablename__ = "kg_concepts"
    __table_args__ = (UniqueConstraint("game_id", "name", name="uq_concept_name_per_game"),)

    game_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("games.id"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    semantic_type: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("kg_concepts.id"),
        nullable=True,
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("sessions.id"),
        nullable=True,
    )

    # Relationships
    game: Mapped[Game] = relationship(foreign_keys=[game_id])
    canonical: Mapped[KGConcept | None] = relationship(
        remote_side="KGConcept.id",
        foreign_keys=[canonical_id],
    )
    session: Mapped[Session | None] = relationship(foreign_keys=[session_id])
    sources: Mapped[list[KGConceptSource]] = relationship(
        back_populates="concept",
        cascade="all, delete-orphan",
    )
    embeddings: Mapped[list[ConceptEmbedding]] = relationship(
        back_populates="concept",
        cascade="all, delete-orphan",
    )


class KGConceptSource(MappedBase):
    """Source traceability for a concept. Links concept to originating chunk.

    Each concept can have multiple sources (cross-document). This is the
    core of AC-504 (full source traceability). Append-only: has ``created_at``
    but no ``updated_at``.

    Attributes:
        concept_id: FK to the concept this source contributes to.
        document_id: FK to the source document. Enables document removal.
        version_id: FK to the specific version. Enables version replacement.
        chunk_id: FK to the source chunk. Enables full traceability chain.
        expansion_id: Expansion association. NULL for base-game documents.
        document_type: Cached from document for conflict resolution priority.
        document_name: Cached for display without joining documents table.
        section_path: Hierarchical section path.
        page_number: Page in original document.
        source_text: Exact text excerpt. The citable foundation.
        is_authoritative: TRUE if highest-priority source for a conflict.
    """

    __tablename__ = "kg_concept_sources"
    __table_args__ = (UniqueConstraint("concept_id", "chunk_id", name="uq_concept_source"),)

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    concept_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("kg_concepts.id", ondelete="CASCADE"),
        nullable=False,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id"),
        nullable=False,
    )
    version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document_versions.id"),
        nullable=False,
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document_chunks.id"),
        nullable=False,
    )
    expansion_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("expansions.id"),
        nullable=True,
    )
    document_type: Mapped[str] = mapped_column(Text, nullable=False)
    document_name: Mapped[str] = mapped_column(Text, nullable=False)
    section_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    is_authoritative: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("TRUE"),
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    concept: Mapped[KGConcept] = relationship(back_populates="sources")
    document: Mapped[Document] = relationship(foreign_keys=[document_id])
    version: Mapped[DocumentVersion] = relationship(foreign_keys=[version_id])
    chunk: Mapped[DocumentChunk] = relationship(foreign_keys=[chunk_id])
    expansion: Mapped[Expansion | None] = relationship(foreign_keys=[expansion_id])


class KGAssociation(Base):
    """Relationship between two concepts in the knowledge graph.

    Follows the standard Base pattern (id + created_at + updated_at).
    Associations are the edges of the knowledge graph.

    Attributes:
        game_id: FK to the owning game.
        source_concept_id: Source end of the directed relationship.
        target_concept_id: Target end of the directed relationship.
        relationship_label: Semantic label (e.g., "modifies", "requires").
        description: Brief description of the relationship.
    """

    __tablename__ = "kg_associations"
    __table_args__ = (
        UniqueConstraint(
            "game_id",
            "source_concept_id",
            "target_concept_id",
            "relationship_label",
            name="uq_association",
        ),
    )

    game_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("games.id"),
        nullable=False,
    )
    source_concept_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("kg_concepts.id", ondelete="CASCADE"),
        nullable=False,
    )
    target_concept_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("kg_concepts.id", ondelete="CASCADE"),
        nullable=False,
    )
    relationship_label: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    game: Mapped[Game] = relationship(foreign_keys=[game_id])
    source_concept: Mapped[KGConcept] = relationship(
        foreign_keys=[source_concept_id],
    )
    target_concept: Mapped[KGConcept] = relationship(
        foreign_keys=[target_concept_id],
    )
    sources: Mapped[list[KGAssociationSource]] = relationship(
        back_populates="association",
        cascade="all, delete-orphan",
    )


class KGAssociationSource(MappedBase):
    """Source traceability for an association. Append-only.

    Parallel structure to KGConceptSource but for associations.

    Attributes:
        association_id: FK to the association this source supports.
        document_id: FK to the source document.
        chunk_id: FK to the source chunk.
        expansion_id: Expansion association. NULL for base-game documents.
        source_text: Exact text excerpt supporting this association.
    """

    __tablename__ = "kg_association_sources"
    __table_args__ = (UniqueConstraint("association_id", "chunk_id", name="uq_assoc_source"),)

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    association_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("kg_associations.id", ondelete="CASCADE"),
        nullable=False,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id"),
        nullable=False,
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document_chunks.id"),
        nullable=False,
    )
    expansion_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("expansions.id"),
        nullable=True,
    )
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    association: Mapped[KGAssociation] = relationship(back_populates="sources")
    document: Mapped[Document] = relationship(foreign_keys=[document_id])
    chunk: Mapped[DocumentChunk] = relationship(foreign_keys=[chunk_id])
    expansion: Mapped[Expansion | None] = relationship(foreign_keys=[expansion_id])


class ConceptEmbedding(MappedBase):
    """Vector embedding for semantic search on a concept.

    Uses pgvector for 1536-dimensional embeddings (OpenAI text-embedding-3-small).
    One embedding per concept (1:1). Append-only: has ``created_at`` only.

    Attributes:
        concept_id: FK to the concept. One embedding per concept.
        embedding: 1536-dimensional vector for cosine similarity search.
        model_id: Embedding model identifier for re-embedding tracking.
    """

    __tablename__ = "concept_embeddings"
    __table_args__ = (UniqueConstraint("concept_id", name="uq_concept_embedding"),)

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    concept_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("kg_concepts.id", ondelete="CASCADE"),
        nullable=False,
    )
    embedding: Mapped[Any] = mapped_column(
        Vector(1536),
        nullable=False,
    )
    model_id: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    concept: Mapped[KGConcept] = relationship(back_populates="embeddings")


class KGAuditLog(MappedBase):
    """Audit trail for all KG operations (AC-509).

    Append-only log of KG construction and retrieval events.
    Has ``created_at`` only — no ``updated_at``.

    Attributes:
        game_id: FK to the game this event relates to.
        event_type: Event classification (e.g., "extraction_complete").
        document_id: FK to the related document (nullable).
        concept_id: UUID of the related concept (nullable, no FK —
            concept may be deleted while audit log persists).
        details: JSONB for event-specific structured data.
    """

    __tablename__ = "kg_audit_log"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    game_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("games.id"),
        nullable=False,
    )
    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("documents.id"),
        nullable=True,
    )
    concept_id: Mapped[uuid.UUID | None] = mapped_column(
        nullable=True,
    )
    details: Mapped[dict[str, Any]] = mapped_column(
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
    game: Mapped[Game] = relationship(foreign_keys=[game_id])
    document: Mapped[Document | None] = relationship(foreign_keys=[document_id])
