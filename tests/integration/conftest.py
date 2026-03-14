"""Integration test fixtures for PostgreSQL.

In CI: uses the PostgreSQL service provided by GitHub Actions (DATABASE_URL env var).
Locally: uses testcontainers to start a PostgreSQL container.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator, Generator

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

logger = logging.getLogger(__name__)


def _get_sync_url() -> str:
    """Return a sync PostgreSQL URL for Alembic migrations."""
    env_url = os.environ.get("DATABASE_URL")
    if env_url:
        return env_url
    # Fallback: testcontainers will set this via the container fixture
    msg = "DATABASE_URL not set and no testcontainer available"
    raise RuntimeError(msg)


def _get_async_url() -> str:
    """Return an async PostgreSQL URL for SQLAlchemy."""
    env_url = os.environ.get("DATABASE_URL_ASYNC")
    if env_url:
        return env_url
    sync_url = _get_sync_url()
    return sync_url.replace("postgresql://", "postgresql+asyncpg://")


@pytest.fixture(scope="session")
def _sync_db_url() -> Generator[str, None, None]:
    """Provide a sync database URL, starting testcontainers if needed."""
    env_url = os.environ.get("DATABASE_URL")
    if env_url:
        yield env_url
        return

    # Local development: use testcontainers
    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:
        pytest.skip("testcontainers not available and DATABASE_URL not set")
        return

    container = PostgresContainer(
        image="postgres:16-alpine",
        username="test",
        password="test",
        dbname="tabletop_oracle_test",
    )
    with container:
        yield container.get_connection_url()


@pytest.fixture(scope="session")
def _async_db_url(_sync_db_url: str) -> str:
    """Convert sync URL to asyncpg URL."""
    env_url = os.environ.get("DATABASE_URL_ASYNC")
    if env_url:
        return env_url
    return _sync_db_url.replace("postgresql://", "postgresql+asyncpg://").replace(
        "psycopg2", "asyncpg"
    )


@pytest.fixture(scope="session")
def _run_migrations(_sync_db_url: str) -> None:
    """Run Alembic migrations against the test database."""
    alembic_cfg = Config()
    alembic_cfg.set_main_option("sqlalchemy.url", _sync_db_url)
    alembic_cfg.set_main_option("script_location", "migrations")
    alembic_cfg.attributes["configure_logger"] = False
    logging.getLogger("alembic").setLevel(logging.WARNING)
    command.upgrade(alembic_cfg, "head")


@pytest.fixture(scope="session")
def sync_db_url(_sync_db_url: str) -> str:
    """Expose sync URL for tests that need direct Alembic access."""
    return _sync_db_url


@pytest.fixture
async def db_session(
    _async_db_url: str,
    _run_migrations: None,
) -> AsyncGenerator[AsyncSession, None]:
    """Yield a clean async session per test with automatic rollback.

    Each test runs inside a SAVEPOINT. After the test, the savepoint
    is rolled back, ensuring complete isolation without re-running
    migrations between tests.
    """
    engine = create_async_engine(_async_db_url, echo=False)
    try:
        async with engine.connect() as conn:
            await conn.begin()
            session_factory = async_sessionmaker(
                bind=conn, class_=AsyncSession, expire_on_commit=False
            )
            session = session_factory()

            await conn.begin_nested()

            @event.listens_for(session.sync_session, "after_transaction_end")
            def restart_savepoint(session_inner: object, transaction: object) -> None:
                """Re-open a nested savepoint after each commit inside the test."""
                if (
                    conn.sync_connection is not None
                    and not conn.sync_connection.in_nested_transaction()
                ):
                    conn.sync_connection.begin_nested()

            yield session

            await session.close()
            await conn.rollback()
    finally:
        await engine.dispose()
