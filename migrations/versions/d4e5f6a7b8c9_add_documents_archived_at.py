"""Add archived_at column to documents table for soft-delete.

Consistent with the F001 soft-delete pattern used by games and
expansions. Includes a partial index for efficient filtering of
non-archived documents in list queries.

Design reference: docs/architecture/epic-002-document-ingestion/design.md
Epic: themachinagod/tabletop-oracle-docs#9
Task: themachinagod/tabletop-oracle-docs#28

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-03-15
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d4e5f6a7b8c9"
down_revision: str | None = "c3d4e5f6a7b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add archived_at column and partial index to documents table."""
    op.add_column(
        "documents",
        sa.Column("archived_at", sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.execute(
        sa.text(
            "CREATE INDEX idx_documents_archived "
            "ON documents (archived_at) WHERE archived_at IS NULL"
        )
    )


def downgrade() -> None:
    """Remove archived_at column and its index from documents table."""
    op.execute(sa.text("DROP INDEX IF EXISTS idx_documents_archived"))
    op.drop_column("documents", "archived_at")
