"""Processing log ORM model.

An append-only audit trail of every action the background worker (and
related services) perform against a capture event — e.g. download
started, prediction saved, storage cleanup, or an error. Useful for
debugging the pipeline and for operational observability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, BigInteger, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, BigIntPK, TimestampMixin

if TYPE_CHECKING:
    from app.models.capture_event import CaptureEvent


class ProcessingLog(TimestampMixin, Base):
    """A single logged action taken by the worker/pipeline for a capture event."""

    __tablename__ = "sc_processing_log"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)

    # Nullable: some log entries (e.g. worker cycle start/stop) are not
    # tied to a specific capture event.
    capture_event_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("sc_capture_event.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    action: Mapped[str] = mapped_column(String(64), nullable=False)
    level: Mapped[str] = mapped_column(String(16), nullable=False, default="INFO")
    message: Mapped[str] = mapped_column(Text, nullable=False)
    detail: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    capture_event: Mapped[CaptureEvent | None] = relationship(back_populates="processing_logs")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<ProcessingLog id={self.id} action={self.action} level={self.level}>"
