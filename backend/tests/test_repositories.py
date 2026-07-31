"""Repository layer tests (against an isolated in-memory SQLite database)."""

from __future__ import annotations

import uuid as uuid_pkg
from datetime import UTC, datetime

import pytest

from app.models.capture_event import CaptureEvent
from app.models.enums import CaptureStatus
from app.models.prediction import Prediction
from app.repositories.capture_event_repository import CaptureEventRepository
from app.repositories.prediction_repository import PredictionRepository
from app.repositories.processing_log_repository import ProcessingLogRepository


def _make_event(device_id: str = "device-1") -> CaptureEvent:
    return CaptureEvent(
        device_id=device_id,
        capture_time=datetime.now(UTC),
        latitude=10.0,
        longitude=20.0,
        heading=90.0,
        speed=5.0,
        image_path=f"{device_id}/image.jpg",
        thumbnail_path=f"{device_id}/thumbnail.jpg",
    )


@pytest.mark.asyncio
async def test_create_and_get_by_uuid(db_session):
    repo = CaptureEventRepository(db_session)
    event = await repo.create(_make_event())
    await db_session.commit()

    fetched = await repo.get_by_uuid(event.uuid)
    assert fetched is not None
    assert fetched.device_id == "device-1"
    assert fetched.status == CaptureStatus.NEW


@pytest.mark.asyncio
async def test_get_by_uuid_returns_none_when_missing(db_session):
    repo = CaptureEventRepository(db_session)
    assert await repo.get_by_uuid(uuid_pkg.uuid4()) is None


@pytest.mark.asyncio
async def test_list_paginated_with_status_filter(db_session):
    repo = CaptureEventRepository(db_session)
    event_a = await repo.create(_make_event("device-a"))
    event_b = await repo.create(_make_event("device-b"))
    event_b.status = CaptureStatus.DONE
    await db_session.flush()
    await db_session.commit()

    all_items, total = await repo.list_paginated(limit=10, offset=0)
    assert total == 2
    assert {item.id for item in all_items} == {event_a.id, event_b.id}

    new_items, new_total = await repo.list_paginated(limit=10, offset=0, status=CaptureStatus.NEW)
    assert new_total == 1
    assert new_items[0].id == event_a.id


@pytest.mark.asyncio
async def test_claim_new_for_processing_transitions_status(db_session):
    repo = CaptureEventRepository(db_session)
    event = await repo.create(_make_event())
    await db_session.commit()

    claimed = await repo.claim_new_for_processing(batch_size=5)
    await db_session.commit()

    assert len(claimed) == 1
    assert claimed[0].id == event.id
    assert claimed[0].status == CaptureStatus.DOWNLOADING

    # A second claim should find nothing left in NEW.
    claimed_again = await repo.claim_new_for_processing(batch_size=5)
    assert claimed_again == []


@pytest.mark.asyncio
async def test_prediction_repository_is_append_only(db_session):
    event_repo = CaptureEventRepository(db_session)
    prediction_repo = PredictionRepository(db_session)

    event = await event_repo.create(_make_event())
    await db_session.commit()

    first = await prediction_repo.create(
        Prediction(
            capture_event_id=event.id,
            model_name="yolo",
            model_version="v1",
            traffic_sign_class="stop_sign",
            confidence=0.8,
        )
    )
    second = await prediction_repo.create(
        Prediction(
            capture_event_id=event.id,
            model_name="yolo",
            model_version="v2",
            traffic_sign_class="stop_sign",
            confidence=0.9,
        )
    )
    await db_session.commit()

    predictions = await prediction_repo.list_for_event(event.id)
    assert len(predictions) == 2
    assert {p.id for p in predictions} == {first.id, second.id}
    # Newest first.
    assert predictions[0].model_version == "v2"


@pytest.mark.asyncio
async def test_processing_log_repository_add(db_session):
    event_repo = CaptureEventRepository(db_session)
    log_repo = ProcessingLogRepository(db_session)

    event = await event_repo.create(_make_event())
    await db_session.commit()

    log = await log_repo.add(
        action="download",
        message="Downloaded image.",
        capture_event_id=event.id,
        detail={"path": "device-1/image.jpg"},
    )
    await db_session.commit()

    assert log.id is not None
    assert log.level == "INFO"
    assert log.detail == {"path": "device-1/image.jpg"}
