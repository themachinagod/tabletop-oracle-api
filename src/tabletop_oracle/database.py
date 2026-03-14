"""Async SQLAlchemy engine and session factory configuration."""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from tabletop_oracle.config import settings

engine = create_async_engine(settings.database_url_async, echo=False)
async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
