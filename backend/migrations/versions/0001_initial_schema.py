"""initial schema: capture_event, prediction, processing_log

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-07-31

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

capture_status_enum = postgresql.ENUM(
    "NEW",
    "DOWNLOADING",
    "PROCESSING",
    "DONE",
    "REJECTED",
    "ERROR",
    name="capture_status",
    create_type=False,
)


def upgrade() -> None:
    capture_status_enum.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "capture_event",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("uuid", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("device_id", sa.String(length=128), nullable=False),
        sa.Column("capture_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=True),
        sa.Column("longitude", sa.Float(), nullable=True),
        sa.Column("heading", sa.Float(), nullable=True),
        sa.Column("speed", sa.Float(), nullable=True),
        sa.Column("image_path", sa.String(length=512), nullable=False),
        sa.Column("thumbnail_path", sa.String(length=512), nullable=False),
        sa.Column("status", capture_status_enum, nullable=False, server_default="NEW"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_capture_event")),
        sa.UniqueConstraint("uuid", name=op.f("uq_capture_event_uuid")),
    )
    op.create_index(op.f("ix_capture_event_device_id"), "capture_event", ["device_id"])
    op.create_index(op.f("ix_capture_event_status"), "capture_event", ["status"])
    op.create_index("ix_capture_event_status_created_at", "capture_event", ["status", "created_at"])

    op.create_table(
        "prediction",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("capture_event_id", sa.BigInteger(), nullable=False),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("traffic_sign_class", sa.String(length=128), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("ocr_text", sa.Text(), nullable=True),
        sa.Column("validation_score", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["capture_event_id"],
            ["capture_event.id"],
            name=op.f("fk_prediction_capture_event_id_capture_event"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_prediction")),
    )
    op.create_index(op.f("ix_prediction_capture_event_id"), "prediction", ["capture_event_id"])

    op.create_table(
        "processing_log",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("capture_event_id", sa.BigInteger(), nullable=True),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("level", sa.String(length=16), nullable=False, server_default="INFO"),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("detail", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["capture_event_id"],
            ["capture_event.id"],
            name=op.f("fk_processing_log_capture_event_id_capture_event"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_processing_log")),
    )
    op.create_index(op.f("ix_processing_log_capture_event_id"), "processing_log", ["capture_event_id"])


def downgrade() -> None:
    op.drop_index(op.f("ix_processing_log_capture_event_id"), table_name="processing_log")
    op.drop_table("processing_log")

    op.drop_index(op.f("ix_prediction_capture_event_id"), table_name="prediction")
    op.drop_table("prediction")

    op.drop_index("ix_capture_event_status_created_at", table_name="capture_event")
    op.drop_index(op.f("ix_capture_event_status"), table_name="capture_event")
    op.drop_index(op.f("ix_capture_event_device_id"), table_name="capture_event")
    op.drop_table("capture_event")

    capture_status_enum.drop(op.get_bind(), checkfirst=True)
