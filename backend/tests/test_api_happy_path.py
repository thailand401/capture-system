"""End-to-end happy-path smoke tests for every public API endpoint.

Complements the more detailed edge-case tests in ``test_api_events.py``:
this module is a single, readable place to see each endpoint's success
path, plus one full-lifecycle test that also drives the background
worker so a client's GET after processing reflects a completed
prediction. No error/validation cases are covered here by design.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import pytest

import app.db.session as db_session_module
from app.repositories.capture_event_repository import CaptureEventRepository
from app.workers.processor import EventProcessor
from tests.fakes import FakePipeline


def _valid_metadata_bytes(**overrides) -> bytes:
    payload = {
        "device_id": "pixel-7-abc123",
        "capture_time": datetime.now(UTC).isoformat(),
        "latitude": 13.7563,
        "longitude": 100.5018,
        "heading": 45.0,
        "speed": 12.5,
    }
    payload.update(overrides)
    return json.dumps(payload).encode("utf-8")


def _multipart_files(metadata_bytes: bytes | None = None) -> dict:
    return {
        "metadata": ("metadata.json", metadata_bytes or _valid_metadata_bytes(), "application/json"),
        "image": ("image.jpg", b"fake-image-bytes", "image/jpeg"),
        "thumbnail": ("thumbnail.jpg", b"fake-thumbnail-bytes", "image/jpeg"),
    }


@pytest.mark.asyncio
async def test_health_check_happy_path(client):
    """GET /health reports liveness without touching the DB or storage."""
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_create_event_happy_path(client, fake_storage):
    """POST /events accepts a well-formed upload and creates a NEW event."""
    response = await client.post("/events", files=_multipart_files())

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "NEW"
    uuid.UUID(body["id"])  # "id" is a valid UUID string
    assert len(fake_storage.objects) == 2  # image + thumbnail uploaded


@pytest.mark.asyncio
async def test_get_event_happy_path(client):
    """GET /events/{id} returns the freshly created event's full detail."""
    create_response = await client.post("/events", files=_multipart_files())
    event_id = create_response.json()["id"]

    response = await client.get(f"/events/{event_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == event_id
    assert body["status"] == "NEW"
    assert body["device_id"] == "pixel-7-abc123"
    assert body["latitude"] == 13.7563
    assert body["longitude"] == 100.5018
    assert body["predictions"] == []


@pytest.mark.asyncio
async def test_list_events_happy_path(client):
    """GET /events returns every created event in a paginated envelope."""
    await client.post("/events", files=_multipart_files(_valid_metadata_bytes(device_id="device-a")))
    await client.post("/events", files=_multipart_files(_valid_metadata_bytes(device_id="device-b")))

    response = await client.get("/events")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert {item["device_id"] for item in body["items"]} == {"device-a", "device-b"}
    assert body["offset"] == 0


@pytest.mark.asyncio
async def test_full_capture_lifecycle_happy_path(client, fake_storage, session_factory, monkeypatch):
    """Create -> worker processes it -> client sees DONE plus a prediction.

    Exercises every layer of the happy path in one flow: the upload API,
    storage, DB persistence, the AI pipeline, and the background worker,
    all surfaced back through the read API exactly as a real client would
    observe them.
    """
    monkeypatch.setattr(db_session_module, "AsyncSessionLocal", session_factory)

    create_response = await client.post("/events", files=_multipart_files())
    event_id = create_response.json()["id"]

    async with session_factory() as session:
        event = await CaptureEventRepository(session).get_by_uuid(uuid.UUID(event_id))
        internal_id = event.id

    processor = EventProcessor(storage=fake_storage, pipeline=FakePipeline())
    await processor.process(internal_id)

    response = await client.get(f"/events/{event_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "DONE"
    assert len(body["predictions"]) == 1
    prediction = body["predictions"][0]
    assert prediction["traffic_sign_class"] == "stop_sign"
    assert prediction["confidence"] == 0.95
    assert prediction["ocr_text"] == "STOP"

    # Temporary storage objects were cleaned up once processing completed.
    assert fake_storage.objects == {}
