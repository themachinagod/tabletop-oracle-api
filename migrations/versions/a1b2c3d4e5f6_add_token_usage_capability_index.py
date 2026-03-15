"""Add capability index to token_usage_log.

Creates ``idx_token_usage_capability`` on ``token_usage_log(capability)``
to support efficient aggregation queries in the UsageAggregationService.
Index creation is non-blocking in PostgreSQL (no table lock).

Design reference: docs/architecture/epic-005-ai-model-config/design.md
Epic: themachinagod/tabletop-oracle-docs#12
Task: themachinagod/tabletop-oracle-api#73

Revision ID: a1b2c3d4e5f6
Revises: f6a7b8c9d0e1
Create Date: 2026-03-15
"""

from collections.abc import Sequence

from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: str | None = "f6a7b8c9d0e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add capability index to token_usage_log."""
    op.create_index(
        "idx_token_usage_capability",
        "token_usage_log",
        ["capability"],
    )


def downgrade() -> None:
    """Remove capability index from token_usage_log."""
    op.drop_index("idx_token_usage_capability", table_name="token_usage_log")
