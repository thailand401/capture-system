"""Shared pytest fixtures for the test suite.

Sets required environment variables *before* importing any application
module (``Settings()`` requires them), then provides a fast, isolated
SQLite engine for repository/worker tests and an httpx test client (with
the database/storage dependencies overridden) for API tests.
"""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "test-service-role-key")
os.environ.setdefault("WORKER_ENABLED", "false")
os.environ.setdefault("LOCAL_CACHE", "./cache")
os.environ.setdefault("LOG_JSON", "false")

from collections.abc import AsyncIterator  # noqa: E402

import pytest_asyncio  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import app.models  # noqa: E402, F401 - populate Base.metadata
from app.api.deps import get_storage_backend  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.db.session import get_db_session  # noqa: E402
from app.main import app  # noqa: E402
from tests.fakes import FakeStorageBackend  # noqa: E402


@pytest_asyncio.fixture
async def test_engine():
    """A fresh SQLite engine (single shared in-memory DB) per test."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def session_factory(test_engine):
    """A sessionmaker bound to the per-test engine."""
    return async_sessionmaker(bind=test_engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def db_session(session_factory) -> AsyncIterator[AsyncSession]:
    """A single async session for direct repository tests."""
    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def fake_storage() -> FakeStorageBackend:
    """A fresh in-memory storage backend double."""
    return FakeStorageBackend()


@pytest_asyncio.fixture
async def client(session_factory, fake_storage) -> AsyncIterator[AsyncClient]:
    """An httpx client wired to the FastAPI app with DB/storage overridden."""

    async def _override_get_db_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db_session] = _override_get_db_session
    app.dependency_overrides[get_storage_backend] = lambda: fake_storage

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()
