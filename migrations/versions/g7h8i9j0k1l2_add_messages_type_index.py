"""Add type index to messages table.

Creates ``idx_messages_type`` on ``messages(type)`` to support efficient
aggregation queries in the QualityMetricsService that filter by message
type (ai_answer, user_question, ai_clarification).

Design reference: docs/architecture/epic-005-ai-model-config/design.md
Epic: themachinagod/tabletop-oracle-docs#12
Task: themachinagod/tabletop-oracle-docs#74

Revision ID: g7h8i9j0k1l2
Revises: f6a7b8c9d0e1
Create Date: 2026-03-15
"""

from collections.abc import Sequence

from alembic import op

revision: str = "g7h8i9j0k1l2"
down_revision: str | None = "f6a7b8c9d0e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add type index to messages."""
    op.create_index(
        "idx_messages_type",
        "messages",
        ["type"],
    )


def downgrade() -> None:
    """Remove type index from messages."""
    op.drop_index("idx_messages_type", table_name="messages")
