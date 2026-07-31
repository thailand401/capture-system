"""Pydantic schemas for capture events.

These models define the public API contract and are intentionally
decoupled from the ORM models — only fields that are safe/useful to
expose externally are included. Internal storage paths and the internal
bigint primary key are never returned to clients; the client-facing "id"
is always the event's ``uuid``.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import CaptureStatus
from app.schemas.prediction import PredictionResponse


class EventMetadata(BaseModel):
    """Shape of the ``metadata`` part of a ``POST /events`` upload."""

    model_config = ConfigDict(extra="forbid")

    uuid: UUID | None = Field(
        default=None,
        description="Optional client-generated idempotency key. Generated server-side if omitted.",
    )
    device_id: str = Field(min_length=1, max_length=128)
    capture_time: datetime = Field(description="Timestamp the image was captured on-device (UTC).")
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    heading: float | None = Field(default=None, ge=0, le=360)
    speed: float | None = Field(default=None, ge=0)


class EventCreateResponse(BaseModel):
    """Response returned immediately after a successful upload."""

    id: UUID
    status: CaptureStatus


class EventResponse(BaseModel):
    """Full status/detail view of a capture event."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: UUID = Field(validation_alias="uuid")
    device_id: str
    capture_time: datetime
    latitude: float | None
    longitude: float | None
    heading: float | None
    speed: float | None
    status: CaptureStatus
    created_at: datetime
    predictions: list[PredictionResponse] = Field(default_factory=list)


class EventSummary(BaseModel):
    """Lightweight representation of a capture event for list views."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: UUID = Field(validation_alias="uuid")
    device_id: str
    capture_time: datetime
    status: CaptureStatus
    created_at: datetime


class EventListResponse(BaseModel):
    """Paginated list of capture events."""

    items: list[EventSummary]
    total: int
    limit: int
    offset: int
