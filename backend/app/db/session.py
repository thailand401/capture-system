"""Async SQLAlchemy engine and session factory.

A single ``AsyncEngine`` is created for the lifetime of the process and
reused everywhere. FastAPI routes obtain a request-scoped session via the
``get_db_session`` dependency; the background worker obtains one via the
``session_scope()`` context manager.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings

settings = get_settings()

# pool_size/max_overflow only apply to QueuePool-based engines (e.g.
# production Postgres via asyncpg). SQLite's default pool (used in tests)
# does not accept them, so they're added conditionally to keep the engine
# portable between production and test database URLs.
_engine_kwargs: dict[str, object] = {"echo": settings.debug, "pool_pre_ping": True}
if not settings.database_url.startswith("sqlite"):
    _engine_kwargs["pool_size"] = 5
    _engine_kwargs["max_overflow"] = 10

engine: AsyncEngine = create_async_engine(settings.database_url, **_engine_kwargs)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding a request-scoped, transactional async session.

    Commits once the request handler completes successfully; rolls back if
    an exception propagates. Handlers/services should rely on repository
    ``flush()`` calls rather than committing directly.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Context manager for a transactional unit of work outside of FastAPI DI.

    Used by the background worker. Commits on success, rolls back on error.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
