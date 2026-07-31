"""Data-access layer for ``CaptureEvent`` aggregates.

All direct SQLAlchemy query construction for capture events lives here.
Services and API routes must go through this repository rather than
issuing queries themselves, keeping persistence concerns isolated and easy
to test/replace.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.models.capture_event import CaptureEvent
from app.models.enums import CaptureStatus
from app.repositories.base import BaseRepository


class CaptureEventRepository(BaseRepository):
    """Persistence operations for ``CaptureEvent`` rows."""

    async def create(self, event: CaptureEvent) -> CaptureEvent:
        """Persist a new capture event and flush to obtain its generated id."""
        self.session.add(event)
        await self.session.flush()
        return event

    async def get_by_uuid(self, event_uuid: UUID) -> CaptureEvent | None:
        """Fetch a single event (with predictions eager-loaded) by its public uuid."""
        stmt = (
            select(CaptureEvent)
            .options(selectinload(CaptureEvent.predictions))
            .where(CaptureEvent.uuid == event_uuid)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_id(self, event_id: int) -> CaptureEvent | None:
        """Fetch a single event by its internal bigint id (worker-internal use)."""
        return await self.session.get(CaptureEvent, event_id)

    async def list_paginated(
        self,
        *,
        limit: int,
        offset: int,
        status: CaptureStatus | None = None,
    ) -> tuple[list[CaptureEvent], int]:
        """Return a page of events (newest first) plus the total matching count."""
        filters = []
        if status is not None:
            filters.append(CaptureEvent.status == status)

        count_stmt = select(func.count()).select_from(CaptureEvent).where(*filters)
        total = (await self.session.execute(count_stmt)).scalar_one()

        list_stmt = (
            select(CaptureEvent)
            .where(*filters)
            .order_by(CaptureEvent.created_at.desc(), CaptureEvent.id.desc())
            .limit(limit)
            .offset(offset)
        )
        items = list((await self.session.execute(list_stmt)).scalars().all())
        return items, total

    async def claim_new_for_processing(self, batch_size: int) -> list[CaptureEvent]:
        """Atomically claim up to ``batch_size`` NEW events for the worker.

        Uses ``SELECT ... FOR UPDATE SKIP LOCKED`` so that, if multiple
        worker instances ever run concurrently, each event is claimed by
        exactly one worker and none block waiting on another's lock.
        """
        stmt = (
            select(CaptureEvent)
            .where(CaptureEvent.status == CaptureStatus.NEW)
            .order_by(CaptureEvent.created_at)
            .limit(batch_size)
            .with_for_update(skip_locked=True)
        )
        result = await self.session.execute(stmt)
        events = list(result.scalars().all())
        for event in events:
            event.status = CaptureStatus.DOWNLOADING
        await self.session.flush()
        return events

    async def update_status(self, event: CaptureEvent, status: CaptureStatus) -> CaptureEvent:
        """Transition an event to a new status."""
        event.status = status
        await self.session.flush()
        return event
