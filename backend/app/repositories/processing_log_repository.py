"""Data-access layer for ``ProcessingLog`` rows (append-only audit trail)."""

from __future__ import annotations

from typing import Any

from app.models.processing_log import ProcessingLog
from app.repositories.base import BaseRepository


class ProcessingLogRepository(BaseRepository):
    """Persistence operations for ``ProcessingLog`` rows."""

    async def add(
        self,
        *,
        action: str,
        message: str,
        level: str = "INFO",
        capture_event_id: int | None = None,
        detail: dict[str, Any] | None = None,
    ) -> ProcessingLog:
        """Append a new log entry describing a worker/pipeline action."""
        log = ProcessingLog(
            capture_event_id=capture_event_id,
            action=action,
            level=level,
            message=message,
            detail=detail,
        )
        self.session.add(log)
        await self.session.flush()
        return log
