"""API-level tests for the /events endpoints."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import pytest


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
async def test_create_event_success(client, fake_storage):
    response = await client.post("/events", files=_multipart_files())

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "NEW"
    assert "id" in body

    # Both the image and thumbnail were uploaded to storage.
    assert len(fake_storage.objects) == 2


@pytest.mark.asyncio
async def test_create_event_invalid_json_metadata(client):
    response = await client.post(
        "/events",
        files={
            "metadata": ("metadata.json", b"not-json", "application/json"),
            "image": ("image.jpg", b"fake-image-bytes", "image/jpeg"),
            "thumbnail": ("thumbnail.jpg", b"fake-thumbnail-bytes", "image/jpeg"),
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_event_missing_required_field(client):
    metadata_dict = json.loads(_valid_metadata_bytes())
    del metadata_dict["device_id"]

    response = await client.post(
        "/events",
        files=_multipart_files(json.dumps(metadata_dict).encode("utf-8")),
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_event_rejects_non_image_content_type(client):
    response = await client.post(
        "/events",
        files={
            "metadata": ("metadata.json", _valid_metadata_bytes(), "application/json"),
            "image": ("image.txt", b"not-an-image", "text/plain"),
            "thumbnail": ("thumbnail.jpg", b"fake-thumbnail-bytes", "image/jpeg"),
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_event_not_found(client):
    response = await client.get(f"/events/{uuid.uuid4()}")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_event_success(client):
    create_response = await client.post("/events", files=_multipart_files())
    event_id = create_response.json()["id"]

    get_response = await client.get(f"/events/{event_id}")
    assert get_response.status_code == 200
    body = get_response.json()
    assert body["id"] == event_id
    assert body["device_id"] == "pixel-7-abc123"
    assert body["predictions"] == []


@pytest.mark.asyncio
async def test_list_events_pagination(client):
    for i in range(3):
        await client.post("/events", files=_multipart_files(_valid_metadata_bytes(device_id=f"device-{i}")))

    response = await client.get("/events", params={"limit": 2, "offset": 0})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 3
    assert len(body["items"]) == 2
    assert body["limit"] == 2
    assert body["offset"] == 0
