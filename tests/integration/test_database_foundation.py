"""Integration tests for the database foundation.

Validates that the Base model conventions, Alembic migrations,
and async session infrastructure work against a real PostgreSQL instance.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import TIMESTAMP, Column, text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from testcontainers.postgres import PostgresContainer

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_connection_and_migrations_applied(db_session: AsyncSession) -> None:
    """Database connection works and migrations have been applied."""
    result = await db_session.execute(text("SELECT 1"))
    assert result.scalar() == 1


@pytest.mark.asyncio
async def test_gen_random_uuid_available(db_session: AsyncSession) -> None:
    """PostgreSQL gen_random_uuid() function is available."""
    result = await db_session.execute(text("SELECT gen_random_uuid()"))
    value = result.scalar()
    assert value is not None
    # Verify it parses as a valid UUID
    parsed = uuid.UUID(str(value))
    assert parsed.version == 4


@pytest.mark.asyncio
async def test_alembic_version_table_exists(db_session: AsyncSession) -> None:
    """Alembic version tracking table exists after migration."""
    result = await db_session.execute(
        text(
            "SELECT EXISTS ("
            "  SELECT FROM information_schema.tables"
            "  WHERE table_name = 'alembic_version'"
            ")"
        )
    )
    assert result.scalar() is True


@pytest.mark.asyncio
async def test_alembic_downgrade_and_upgrade(
    postgres_container: PostgresContainer,
) -> None:
    """Alembic downgrade to base and upgrade to head both succeed."""
    from alembic import command
    from alembic.config import Config

    sync_url = postgres_container.get_connection_url()
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", sync_url)
    alembic_cfg.set_main_option("script_location", "migrations")

    # Downgrade to base (empty schema)
    command.downgrade(alembic_cfg, "base")

    # Upgrade back to head
    command.upgrade(alembic_cfg, "head")


def test_base_model_id_has_server_default() -> None:
    """Base.id column uses gen_random_uuid() as server_default."""
    from tabletop_oracle.models.base import Base

    id_col: Column[uuid.UUID] = Base.__dict__["id"].property.columns[0]  # type: ignore[assignment]
    assert id_col.server_default is not None
    assert "gen_random_uuid" in str(id_col.server_default.arg)


def test_base_model_timestamps_use_timestamptz() -> None:
    """Base created_at and updated_at columns use TIMESTAMP WITH TIME ZONE."""
    from tabletop_oracle.models.base import Base

    for col_name in ("created_at", "updated_at"):
        col: Column[object] = Base.__dict__[col_name].property.columns[0]  # type: ignore[assignment]
        assert isinstance(col.type, TIMESTAMP)
        assert col.type.timezone is True


def test_base_model_id_has_no_python_default() -> None:
    """Base.id does not set a Python-side default (server generates UUID)."""
    from tabletop_oracle.models.base import Base

    id_col: Column[uuid.UUID] = Base.__dict__["id"].property.columns[0]  # type: ignore[assignment]
    assert id_col.default is None


def test_base_model_updated_at_has_no_onupdate() -> None:
    """Base.updated_at does not use SQLAlchemy onupdate (trigger-managed)."""
    from tabletop_oracle.models.base import Base

    col: Column[object] = Base.__dict__["updated_at"].property.columns[0]  # type: ignore[assignment]
    assert col.onupdate is None
