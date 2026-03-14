"""Add auth_sessions table for server-side session management.

Creates the auth_sessions table with TEXT primary key (opaque session
token), FK to users(id) with ON DELETE CASCADE, and indexes on user_id
and expires_at.

Design reference: docs/architecture/f002-authentication-authorisation/design.md
ADR reference: docs/architecture/decisions/ADR-003-session-management.md
Epic: themachinagod/tabletop-oracle-docs#5

Revision ID: a1f2b3c4d5e6
Revises: b6c39a30deff
Create Date: 2026-03-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a1f2b3c4d5e6"
down_revision: str | None = "b6c39a30deff"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the auth_sessions table with indexes."""
    op.create_table(
        "auth_sessions",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column(
            "user_id",
            sa.UUID(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "last_active_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "expires_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
        ),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("ip_address", sa.Text(), nullable=True),
    )
    op.create_index("idx_auth_sessions_user", "auth_sessions", ["user_id"])
    op.create_index("idx_auth_sessions_expires", "auth_sessions", ["expires_at"])


def downgrade() -> None:
    """Drop the auth_sessions table and its indexes."""
    op.drop_index("idx_auth_sessions_expires", table_name="auth_sessions")
    op.drop_index("idx_auth_sessions_user", table_name="auth_sessions")
    op.drop_table("auth_sessions")
