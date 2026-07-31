"""Background worker tests (``EventProcessor`` for a single claimed event)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

import app.db.session as db_session_module
from app.core.config import get_settings
from app.models.capture_event import CaptureEvent
from app.models.enums import CaptureStatus
from app.repositories.capture_event_repository import CaptureEventRepository
from app.repositories.prediction_repository import PredictionRepository
from app.workers.processor import EventProcessor
from tests.fakes import FakePipeline, FakeStorageBackend


@pytest.fixture(autouse=True)
def _patch_session_scope(monkeypatch, session_factory):
    """Point the worker's session_scope()/AsyncSessionLocal at the test DB."""
    monkeypatch.setattr(db_session_module, "AsyncSessionLocal", session_factory)


@pytest.fixture
def local_cache_dir(tmp_path, monkeypatch):
    """Redirect Settings.local_cache to a temp dir for the duration of a test."""
    monkeypatch.setenv("LOCAL_CACHE", str(tmp_path))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


async def _seed_new_event(session_factory) -> tuple[int, str, str]:
    async with session_factory() as session:
        repo = CaptureEventRepository(session)
        event = CaptureEvent(
            device_id="device-1",
            capture_time=datetime.now(UTC),
            image_path="device-1/uuid/image.jpg",
            thumbnail_path="device-1/uuid/thumbnail.jpg",
        )
        await repo.create(event)
        await session.commit()
        return event.id, event.image_path, event.thumbnail_path


@pytest.mark.asyncio
async def test_event_processor_success_path(session_factory, local_cache_dir):
    storage = FakeStorageBackend()
    event_id, image_path, thumbnail_path = await _seed_new_event(session_factory)
    storage.objects[image_path] = b"raw-jpeg-bytes"
    storage.objects[thumbnail_path] = b"raw-thumb-bytes"

    processor = EventProcessor(storage=storage, pipeline=FakePipeline())
    await processor.process(event_id)

    async with session_factory() as session:
        event_repo = CaptureEventRepository(session)
        prediction_repo = PredictionRepository(session)

        event = await event_repo.get_by_id(event_id)
        assert event.status == CaptureStatus.DONE

        predictions = await prediction_repo.list_for_event(event_id)
        assert len(predictions) == 1
        assert predictions[0].traffic_sign_class == "stop_sign"

    # Storage objects were deleted, and nothing lingers in the local cache.
    assert storage.objects == {}
    assert list(local_cache_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_event_processor_marks_error_on_download_failure(session_factory, local_cache_dir):
    storage = FakeStorageBackend(fail_download=True)
    event_id, _, _ = await _seed_new_event(session_factory)

    processor = EventProcessor(storage=storage, pipeline=FakePipeline())
    await processor.process(event_id)

    async with session_factory() as session:
        event_repo = CaptureEventRepository(session)
        event = await event_repo.get_by_id(event_id)
        assert event.status == CaptureStatus.ERROR
