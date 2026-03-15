"""Add cancelled_at column and token_usage_log message index.

Adds a nullable TIMESTAMPTZ cancelled_at column to messages for query
cancellation tracking, and an index on token_usage_log(message_id) for
efficient per-query token usage lookups.

Both changes are zero-downtime: nullable column addition and index
creation are non-blocking in PostgreSQL.

Design reference: docs/architecture/epic-004-ai-workflow/design.md
Epic: themachinagod/tabletop-oracle-docs#11
Task: themachinagod/tabletop-oracle-docs#69

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-03-15
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f6a7b8c9d0e1"
down_revision: str | None = "e5f6a7b8c9d0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add cancelled_at column to messages and message index to token_usage_log."""
    op.add_column(
        "messages",
        sa.Column("cancelled_at", sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.create_index(
        "idx_token_usage_message",
        "token_usage_log",
        ["message_id"],
    )


def downgrade() -> None:
    """Remove cancelled_at column and token_usage_log message index."""
    op.drop_index("idx_token_usage_message", table_name="token_usage_log")
    op.drop_column("messages", "cancelled_at")
