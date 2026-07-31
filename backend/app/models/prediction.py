"""Prediction ORM model.

Stores the result of one AI pipeline run against a capture event. Multiple
predictions may exist for the same capture event (e.g. re-processing with
a newer model version) — rows are append-only and never overwritten or
updated in place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, BigIntPK, TimestampMixin

if TYPE_CHECKING:
    from app.models.capture_event import CaptureEvent


class Prediction(TimestampMixin, Base):
    """An AI pipeline result produced for a single capture event."""

    __tablename__ = "prediction"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)

    capture_event_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("capture_event.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)

    traffic_sign_class: Mapped[str | None] = mapped_column(String(128), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    ocr_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    validation_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    capture_event: Mapped[CaptureEvent] = relationship(back_populates="predictions")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<Prediction id={self.id} capture_event_id={self.capture_event_id} "
            f"class={self.traffic_sign_class}>"
        )
