"""Background worker polling loop.

Runs as an asyncio task inside the FastAPI process (started from the app's
lifespan). Every ``WORKER_INTERVAL_SECONDS`` it claims a batch of ``NEW``
capture events and processes each one through the AI pipeline. The loop
itself never raises — any unexpected error is logged and the loop simply
waits for the next cycle, so a single bad cycle can never take the worker
down.
"""

from __future__ import annotations

import asyncio

from app.ai.factory import build_default_pipeline
from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import session_scope
from app.repositories.capture_event_repository import CaptureEventRepository
from app.storage.supabase_storage import SupabaseStorageService
from app.workers.processor import EventProcessor

logger = get_logger(__name__)


async def _run_one_cycle(processor: EventProcessor, batch_size: int) -> int:
    """Claim up to ``batch_size`` NEW events and process each one.

    Returns the number of events claimed (0 if none were pending).
    """
    async with session_scope() as session:
        capture_event_repo = CaptureEventRepository(session)
        claimed_events = await capture_event_repo.claim_new_for_processing(batch_size)
        event_ids = [event.id for event in claimed_events]

    for event_id in event_ids:
        await processor.process(event_id)

    return len(event_ids)


async def worker_loop() -> None:
    """Poll for NEW capture events forever, until the task is cancelled."""
    settings = get_settings()
    storage = SupabaseStorageService()
    pipeline = build_default_pipeline()
    processor = EventProcessor(storage=storage, pipeline=pipeline)

    logger.info("worker_loop_starting", interval_seconds=settings.worker_interval_seconds)

    while True:
        try:
            claimed = await _run_one_cycle(processor, settings.worker_batch_size)
            if claimed:
                logger.info("worker_cycle_completed", claimed=claimed)
        except asyncio.CancelledError:
            logger.info("worker_loop_cancelled")
            raise
        except Exception:  # noqa: BLE001 - the worker loop must never crash
            logger.error("worker_cycle_failed")

        try:
            await asyncio.sleep(settings.worker_interval_seconds)
        except asyncio.CancelledError:
            logger.info("worker_loop_cancelled")
            raise
