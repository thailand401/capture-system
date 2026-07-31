"""HTTP routes for capture events.

Route handlers are intentionally thin: they parse/validate the incoming
HTTP request, delegate all business logic to services/repositories, and
shape the response. No persistence or storage logic lives here.
"""

from __future__ import annotations

import json
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Query, UploadFile, status
from pydantic import ValidationError

from app.api.deps import get_capture_event_repository, get_event_service
from app.core.config import get_settings
from app.core.exceptions import NotFoundError, ValidationAppError
from app.models.enums import CaptureStatus
from app.repositories.capture_event_repository import CaptureEventRepository
from app.schemas.capture_event import (
    EventCreateResponse,
    EventListResponse,
    EventMetadata,
    EventResponse,
    EventSummary,
)
from app.services.event_service import EventService

router = APIRouter(prefix="/events", tags=["events"])

settings = get_settings()


async def _read_validated_image(file: UploadFile, *, field_name: str) -> tuple[bytes, str]:
    """Read an uploaded image part, enforcing a basic content-type/size check."""
    content_type = file.content_type or "application/octet-stream"
    if not content_type.startswith("image/"):
        raise ValidationAppError(f"'{field_name}' must be an image upload, got '{content_type}'.")

    data = await file.read()
    if not data:
        raise ValidationAppError(f"'{field_name}' must not be empty.")
    if len(data) > settings.max_upload_size_bytes:
        raise ValidationAppError(f"'{field_name}' exceeds the maximum allowed size.")
    return data, content_type


@router.post("", response_model=EventCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_event(
    event_service: Annotated[EventService, Depends(get_event_service)],
    metadata: Annotated[UploadFile, File(..., description="metadata.json describing the capture")],
    image: Annotated[UploadFile, File(..., description="image.jpg - the captured photo")],
    thumbnail: Annotated[UploadFile, File(..., description="thumbnail.jpg - a small preview image")],
) -> EventCreateResponse:
    """Ingest a new capture event uploaded by an Android device.

    Accepts a multipart request with three parts: ``metadata`` (JSON),
    ``image`` and ``thumbnail``. The image/thumbnail are uploaded to
    Supabase Storage and a new ``capture_event`` row is created with
    status ``NEW`` for the background worker to pick up.
    """
    raw_metadata = await metadata.read()
    try:
        metadata_dict = json.loads(raw_metadata)
    except json.JSONDecodeError as exc:
        raise ValidationAppError("metadata.json is not valid JSON.") from exc

    try:
        event_metadata = EventMetadata.model_validate(metadata_dict)
    except ValidationError as exc:
        raise ValidationAppError(
            "metadata.json failed validation.", details={"errors": exc.errors()}
        ) from exc

    image_bytes, image_content_type = await _read_validated_image(image, field_name="image")
    thumbnail_bytes, thumbnail_content_type = await _read_validated_image(thumbnail, field_name="thumbnail")

    event = await event_service.create_event(
        metadata=event_metadata,
        image_bytes=image_bytes,
        image_content_type=image_content_type,
        thumbnail_bytes=thumbnail_bytes,
        thumbnail_content_type=thumbnail_content_type,
    )
    return EventCreateResponse(id=event.uuid, status=event.status)


@router.get("/{event_id}", response_model=EventResponse)
async def get_event(
    event_id: UUID,
    capture_event_repo: Annotated[CaptureEventRepository, Depends(get_capture_event_repository)],
) -> EventResponse:
    """Return the current status and predictions for a single capture event."""
    event = await capture_event_repo.get_by_uuid(event_id)
    if event is None:
        raise NotFoundError(f"Capture event '{event_id}' was not found.")
    return EventResponse.model_validate(event)


@router.get("", response_model=EventListResponse)
async def list_events(
    capture_event_repo: Annotated[CaptureEventRepository, Depends(get_capture_event_repository)],
    limit: Annotated[int, Query(ge=1, le=settings.max_page_size)] = settings.default_page_size,
    offset: Annotated[int, Query(ge=0)] = 0,
    status_filter: Annotated[CaptureStatus | None, Query(alias="status")] = None,
) -> EventListResponse:
    """Return a paginated list of capture events, newest first."""
    items, total = await capture_event_repo.list_paginated(limit=limit, offset=offset, status=status_filter)
    return EventListResponse(
        items=[EventSummary.model_validate(item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )
