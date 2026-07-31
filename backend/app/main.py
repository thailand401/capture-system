"""FastAPI application entrypoint.

Wires together configuration, logging, routers, exception handlers, and
the background worker lifecycle.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.workers.runner import worker_loop

settings = get_settings()
configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage startup/shutdown: launch and gracefully stop the background worker."""
    worker_task: asyncio.Task[None] | None = None
    if settings.worker_enabled:
        worker_task = asyncio.create_task(worker_loop(), name="capture-event-worker")
        logger.info("worker_started", interval_seconds=settings.worker_interval_seconds)
    else:
        logger.info("worker_disabled")

    yield

    if worker_task is not None:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        logger.info("worker_stopped")


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    lifespan=lifespan,
)

register_exception_handlers(app)
app.include_router(api_router)
