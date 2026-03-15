"""Add knowledge graph tables for EPIC-003.

Creates 6 tables for the knowledge graph engine: kg_concepts,
kg_concept_sources, kg_associations, kg_association_sources,
concept_embeddings, and kg_audit_log. Enables the pgvector extension
for vector similarity search and creates an HNSW index on
concept_embeddings.

Design reference: docs/architecture/epic-003-knowledge-graph-engine/design.md
Epic: themachinagod/tabletop-oracle-docs#10
Task: themachinagod/tabletop-oracle-docs#45

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-03-15
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e5f6a7b8c9d0"
down_revision: str | None = "d4e5f6a7b8c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Tables that receive the update_updated_at trigger (have updated_at column)
KG_TRIGGER_TABLES = ["kg_concepts", "kg_associations"]

# Tables in reverse dependency order for rollback
KG_TABLES_DROP_ORDER = [
    "kg_audit_log",
    "concept_embeddings",
    "kg_association_sources",
    "kg_associations",
    "kg_concept_sources",
    "kg_concepts",
]


def _enable_pgvector() -> None:
    """Enable the pgvector extension for vector similarity search."""
    op.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))


def _create_kg_concepts() -> None:
    """Create kg_concepts table with indexes."""
    op.create_table(
        "kg_concepts",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "game_id",
            sa.UUID(),
            sa.ForeignKey("games.id"),
            nullable=False,
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("semantic_type", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column(
            "canonical_id",
            sa.UUID(),
            sa.ForeignKey("kg_concepts.id"),
            nullable=True,
        ),
        sa.Column(
            "session_id",
            sa.UUID(),
            sa.ForeignKey("sessions.id"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("game_id", "name", name="uq_concept_name_per_game"),
    )
    op.create_index("idx_kg_concepts_game", "kg_concepts", ["game_id"])
    op.create_index("idx_kg_concepts_type", "kg_concepts", ["game_id", "semantic_type"])
    op.execute(
        sa.text(
            "CREATE INDEX idx_kg_concepts_session ON kg_concepts (session_id) "
            "WHERE session_id IS NOT NULL"
        )
    )


def _create_kg_concept_sources() -> None:
    """Create kg_concept_sources table with indexes."""
    op.create_table(
        "kg_concept_sources",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "concept_id",
            sa.UUID(),
            sa.ForeignKey("kg_concepts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "document_id",
            sa.UUID(),
            sa.ForeignKey("documents.id"),
            nullable=False,
        ),
        sa.Column(
            "version_id",
            sa.UUID(),
            sa.ForeignKey("document_versions.id"),
            nullable=False,
        ),
        sa.Column(
            "chunk_id",
            sa.UUID(),
            sa.ForeignKey("document_chunks.id"),
            nullable=False,
        ),
        sa.Column(
            "expansion_id",
            sa.UUID(),
            sa.ForeignKey("expansions.id"),
            nullable=True,
        ),
        sa.Column("document_type", sa.Text(), nullable=False),
        sa.Column("document_name", sa.Text(), nullable=False),
        sa.Column("section_path", sa.Text(), nullable=True),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("source_text", sa.Text(), nullable=False),
        sa.Column(
            "is_authoritative",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("TRUE"),
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("concept_id", "chunk_id", name="uq_concept_source"),
    )
    op.create_index("idx_concept_sources_concept", "kg_concept_sources", ["concept_id"])
    op.create_index("idx_concept_sources_document", "kg_concept_sources", ["document_id"])
    op.create_index("idx_concept_sources_expansion", "kg_concept_sources", ["expansion_id"])


def _create_kg_associations() -> None:
    """Create kg_associations table with indexes."""
    op.create_table(
        "kg_associations",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "game_id",
            sa.UUID(),
            sa.ForeignKey("games.id"),
            nullable=False,
        ),
        sa.Column(
            "source_concept_id",
            sa.UUID(),
            sa.ForeignKey("kg_concepts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "target_concept_id",
            sa.UUID(),
            sa.ForeignKey("kg_concepts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("relationship_label", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "game_id",
            "source_concept_id",
            "target_concept_id",
            "relationship_label",
            name="uq_association",
        ),
    )
    op.create_index("idx_associations_game", "kg_associations", ["game_id"])
    op.create_index("idx_associations_source", "kg_associations", ["source_concept_id"])
    op.create_index("idx_associations_target", "kg_associations", ["target_concept_id"])


def _create_kg_association_sources() -> None:
    """Create kg_association_sources table with indexes."""
    op.create_table(
        "kg_association_sources",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "association_id",
            sa.UUID(),
            sa.ForeignKey("kg_associations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "document_id",
            sa.UUID(),
            sa.ForeignKey("documents.id"),
            nullable=False,
        ),
        sa.Column(
            "chunk_id",
            sa.UUID(),
            sa.ForeignKey("document_chunks.id"),
            nullable=False,
        ),
        sa.Column(
            "expansion_id",
            sa.UUID(),
            sa.ForeignKey("expansions.id"),
            nullable=True,
        ),
        sa.Column("source_text", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("association_id", "chunk_id", name="uq_assoc_source"),
    )
    op.create_index(
        "idx_assoc_sources_association",
        "kg_association_sources",
        ["association_id"],
    )
    op.create_index(
        "idx_assoc_sources_document",
        "kg_association_sources",
        ["document_id"],
    )


def _create_concept_embeddings() -> None:
    """Create concept_embeddings table with HNSW index."""
    op.create_table(
        "concept_embeddings",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "concept_id",
            sa.UUID(),
            sa.ForeignKey("kg_concepts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # vector(1536) column — raw SQL because Alembic doesn't natively
        # support pgvector types
        sa.Column("model_id", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("concept_id", name="uq_concept_embedding"),
    )
    # Add vector column via raw SQL (pgvector type)
    op.execute(
        sa.text("ALTER TABLE concept_embeddings ADD COLUMN embedding vector(1536) NOT NULL")
    )
    # HNSW index for cosine similarity search
    op.execute(
        sa.text(
            "CREATE INDEX idx_embeddings_cosine ON concept_embeddings "
            "USING hnsw (embedding vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 200)"
        )
    )


def _create_kg_audit_log() -> None:
    """Create kg_audit_log table with indexes."""
    op.create_table(
        "kg_audit_log",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "game_id",
            sa.UUID(),
            sa.ForeignKey("games.id"),
            nullable=False,
        ),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column(
            "document_id",
            sa.UUID(),
            sa.ForeignKey("documents.id"),
            nullable=True,
        ),
        sa.Column("concept_id", sa.UUID(), nullable=True),
        sa.Column(
            "details",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("idx_audit_game", "kg_audit_log", ["game_id"])
    op.create_index("idx_audit_event", "kg_audit_log", ["event_type"])
    op.create_index("idx_audit_created", "kg_audit_log", ["created_at"])


def _apply_kg_triggers() -> None:
    """Apply update_updated_at trigger to KG tables with updated_at."""
    for table in KG_TRIGGER_TABLES:
        op.execute(
            sa.text(f"""
            CREATE TRIGGER trg_{table}_updated_at
                BEFORE UPDATE ON {table}
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at()
        """)
        )


def _drop_kg_triggers() -> None:
    """Drop update_updated_at triggers from KG tables."""
    for table in reversed(KG_TRIGGER_TABLES):
        op.execute(sa.text(f"DROP TRIGGER IF EXISTS trg_{table}_updated_at ON {table}"))


def upgrade() -> None:
    """Add pgvector extension and all knowledge graph tables."""
    _enable_pgvector()
    _create_kg_concepts()
    _create_kg_concept_sources()
    _create_kg_associations()
    _create_kg_association_sources()
    _create_concept_embeddings()
    _create_kg_audit_log()
    _apply_kg_triggers()


def downgrade() -> None:
    """Drop all knowledge graph tables in reverse dependency order."""
    _drop_kg_triggers()

    for table in KG_TABLES_DROP_ORDER:
        op.drop_table(table)

    # Note: We do NOT drop the pgvector extension on downgrade.
    # Other migrations or application code may depend on it, and
    # dropping extensions requires elevated privileges.
