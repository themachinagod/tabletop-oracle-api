"""Integration test fixtures using testcontainers for PostgreSQL.

Provides a real PostgreSQL instance per test session via Docker,
runs Alembic migrations to set up the schema, and yields a clean
async session per test with automatic rollback.
"""

from collections.abc import AsyncGenerator

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from testcontainers.postgres import PostgresContainer


@pytest.fixture(scope="session")
def postgres_container() -> PostgresContainer:
    """Start a PostgreSQL 16 container for the test session.

    The container is started once and reused across all integration tests.
    It is stopped and removed after the session completes.
    """
    container = PostgresContainer(
        image="postgres:16-alpine",
        username="test",
        password="test",
        dbname="tabletop_oracle_test",
    )
    with container:
        yield container


@pytest.fixture(scope="session")
def postgres_url(postgres_container: PostgresContainer) -> str:
    """Return the asyncpg connection URL for the test container."""
    sync_url = postgres_container.get_connection_url()
    return sync_url.replace("psycopg2", "asyncpg")


@pytest.fixture(scope="session")
def _run_migrations(postgres_container: PostgresContainer) -> None:
    """Run Alembic migrations against the test database.

    Uses the sync (psycopg2) URL since Alembic's offline/online runner
    invokes asyncio.run() internally, which conflicts with the event
    loop. Creates Config programmatically (not from alembic.ini) to
    avoid logging configuration issues in test environments.
    """
    import logging

    sync_url = postgres_container.get_connection_url()
    alembic_cfg = Config()
    alembic_cfg.set_main_option("sqlalchemy.url", sync_url)
    alembic_cfg.set_main_option("script_location", "migrations")
    alembic_cfg.attributes["configure_logger"] = False
    logging.getLogger("alembic").setLevel(logging.WARNING)
    command.upgrade(alembic_cfg, "head")


@pytest.fixture(scope="session")
async def async_engine(
    postgres_url: str,
    _run_migrations: None,
) -> AsyncGenerator[AsyncEngine, None]:
    """Create an async engine connected to the test PostgreSQL instance."""
    eng = create_async_engine(postgres_url, echo=False)
    yield eng
    await eng.dispose()


@pytest.fixture
async def db_session(
    async_engine: AsyncEngine,
) -> AsyncGenerator[AsyncSession, None]:
    """Yield a clean async session per test with automatic rollback.

    Each test runs inside a SAVEPOINT. After the test, the savepoint
    is rolled back, ensuring complete isolation without re-running
    migrations between tests.
    """
    async with async_engine.connect() as conn:
        await conn.begin()
        session_factory = async_sessionmaker(
            bind=conn, class_=AsyncSession, expire_on_commit=False
        )
        session = session_factory()

        # Begin a nested savepoint so the test can commit/rollback
        # without affecting the outer transaction
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


@pytest.fixture
async def verify_db_connection(db_session: AsyncSession) -> None:
    """Smoke-test fixture that verifies the database connection works."""
    result = await db_session.execute(text("SELECT 1"))
    assert result.scalar() == 1
