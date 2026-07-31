"""FastAPI dependency providers.

Centralizes how request-scoped repositories, services, and storage
backends are constructed and injected into route handlers.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.repositories.capture_event_repository import CaptureEventRepository
from app.repositories.prediction_repository import PredictionRepository
from app.repositories.processing_log_repository import ProcessingLogRepository
from app.services.event_service import EventService
from app.storage.base import StorageBackend
from app.storage.supabase_storage import SupabaseStorageService

DbSession = Annotated[AsyncSession, Depends(get_db_session)]

# A single storage client instance is reused across requests so the
# underlying Supabase client (and its connection pool) is not recreated
# per request.
_storage_singleton: StorageBackend = SupabaseStorageService()


def get_storage_backend() -> StorageBackend:
    """Provide the active storage backend implementation."""
    return _storage_singleton


def get_capture_event_repository(session: DbSession) -> CaptureEventRepository:
    return CaptureEventRepository(session)


def get_prediction_repository(session: DbSession) -> PredictionRepository:
    return PredictionRepository(session)


def get_processing_log_repository(session: DbSession) -> ProcessingLogRepository:
    return ProcessingLogRepository(session)


def get_event_service(
    capture_event_repo: Annotated[CaptureEventRepository, Depends(get_capture_event_repository)],
    processing_log_repo: Annotated[ProcessingLogRepository, Depends(get_processing_log_repository)],
    storage: Annotated[StorageBackend, Depends(get_storage_backend)],
) -> EventService:
    return EventService(
        storage=storage,
        capture_event_repo=capture_event_repo,
        processing_log_repo=processing_log_repo,
    )
