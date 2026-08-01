"""Capture event ORM model.

Represents a single traffic-sign capture uploaded by an Android device. The
event's image and thumbnail are stored temporarily in Supabase Storage;
only their storage *paths* are persisted here — never the binary image
data.
"""

from __future__ import annotations

import uuid as uuid_pkg
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Enum, Float, Index, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, BigIntPK, TimestampMixin
from app.models.enums import CaptureStatus

if TYPE_CHECKING:
    from app.models.prediction import Prediction
    from app.models.processing_log import ProcessingLog


class CaptureEvent(TimestampMixin, Base):
    """A traffic-sign capture event uploaded from an Android device."""

    __tablename__ = "sc_capture_event"
    __table_args__ = (Index("ix_sc_capture_event_status_created_at", "status", "created_at"),)

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)

    # Client-facing unique identifier (also usable as an idempotency key so
    # a retried upload from an unreliable mobile network does not create a
    # duplicate row). Exposed to the API as "id"; the internal bigint above
    # is never returned to clients.
    uuid: Mapped[uuid_pkg.UUID] = mapped_column(
        Uuid(as_uuid=True),
        unique=True,
        nullable=False,
        default=uuid_pkg.uuid4,
    )

    device_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    # When the Android device captured the image (may differ from
    # created_at, which is when the server received the upload).
    capture_time: Mapped[datetime] = mapped_column(nullable=False)

    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    heading: Mapped[float | None] = mapped_column(Float, nullable=True)
    speed: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Paths inside the Supabase Storage bucket. Never store image binaries
    # in Postgres.
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)
    thumbnail_path: Mapped[str] = mapped_column(String(512), nullable=False)

    status: Mapped[CaptureStatus] = mapped_column(
        Enum(CaptureStatus, name="capture_status", native_enum=True),
        nullable=False,
        default=CaptureStatus.NEW,
        server_default=CaptureStatus.NEW.value,
        index=True,
    )

    predictions: Mapped[list[Prediction]] = relationship(
        back_populates="capture_event",
        cascade="all, delete-orphan",
        order_by="[Prediction.created_at.desc(), Prediction.id.desc()]",
    )
    processing_logs: Mapped[list[ProcessingLog]] = relationship(
        back_populates="capture_event",
        cascade="all, delete-orphan",
        order_by="[ProcessingLog.created_at, ProcessingLog.id]",
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<CaptureEvent id={self.id} uuid={self.uuid} status={self.status}>"
