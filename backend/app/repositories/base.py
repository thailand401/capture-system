"""Base repository providing shared session access for concrete repositories."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession


class BaseRepository:
    """Holds the async session shared by all repository methods."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
