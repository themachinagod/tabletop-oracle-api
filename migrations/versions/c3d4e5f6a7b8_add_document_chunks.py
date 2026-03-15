"""Add document_chunks table for ingestion pipeline output.

Stores structured content chunks produced by the ingestion pipeline's
chunking engine. Chunks are consumed by the KG engine (EPIC-003) and
used for content preview.

Design reference: docs/architecture/epic-002-document-ingestion/design.md
Epic: themachinagod/tabletop-oracle-docs#9
Task: themachinagod/tabletop-oracle-docs#28

Revision ID: c3d4e5f6a7b8
Revises: a1f2b3c4d5e6
Create Date: 2026-03-15
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "c3d4e5f6a7b8"
down_revision: str | None = "a1f2b3c4d5e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the document_chunks table with indexes and constraints."""
    op.create_table(
        "document_chunks",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "document_id",
            sa.UUID(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "version_id",
            sa.UUID(),
            sa.ForeignKey("document_versions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("chunk_type", sa.Text(), nullable=False, server_default="text"),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("section_path", sa.Text(), nullable=True),
        sa.Column("heading", sa.Text(), nullable=True),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("token_estimate", sa.Integer(), nullable=False),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "document_id",
            "version_id",
            "chunk_index",
            name="uq_chunk_index",
        ),
    )
    op.create_index("idx_chunks_document", "document_chunks", ["document_id"])
    op.create_index("idx_chunks_version", "document_chunks", ["version_id"])


def downgrade() -> None:
    """Drop the document_chunks table and its indexes."""
    op.drop_index("idx_chunks_version", table_name="document_chunks")
    op.drop_index("idx_chunks_document", table_name="document_chunks")
    op.drop_table("document_chunks")
