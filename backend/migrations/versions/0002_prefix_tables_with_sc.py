"""prefix existing tables with sc_

Revision ID: 0002_prefix_tables_with_sc
Revises: 0001_initial_schema
Create Date: 2026-08-01

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002_prefix_tables_with_sc"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _has_table(table_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    if not _has_table("capture_event"):
        return

    op.rename_table("capture_event", "sc_capture_event")
    op.rename_table("prediction", "sc_prediction")
    op.rename_table("processing_log", "sc_processing_log")

    op.execute("ALTER INDEX ix_capture_event_device_id RENAME TO ix_sc_capture_event_device_id")
    op.execute("ALTER INDEX ix_capture_event_status RENAME TO ix_sc_capture_event_status")
    op.execute(
        "ALTER INDEX ix_capture_event_status_created_at "
        "RENAME TO ix_sc_capture_event_status_created_at"
    )
    op.execute("ALTER INDEX ix_prediction_capture_event_id RENAME TO ix_sc_prediction_capture_event_id")
    op.execute(
        "ALTER INDEX ix_processing_log_capture_event_id "
        "RENAME TO ix_sc_processing_log_capture_event_id"
    )

    op.execute(
        "ALTER TABLE sc_capture_event "
        "RENAME CONSTRAINT pk_capture_event TO pk_sc_capture_event"
    )
    op.execute(
        "ALTER TABLE sc_capture_event "
        "RENAME CONSTRAINT uq_capture_event_uuid TO uq_sc_capture_event_uuid"
    )
    op.execute(
        "ALTER TABLE sc_prediction "
        "RENAME CONSTRAINT pk_prediction TO pk_sc_prediction"
    )
    op.execute(
        "ALTER TABLE sc_prediction "
        "RENAME CONSTRAINT fk_prediction_capture_event_id_capture_event "
        "TO fk_sc_prediction_capture_event_id_sc_capture_event"
    )
    op.execute(
        "ALTER TABLE sc_processing_log "
        "RENAME CONSTRAINT pk_processing_log TO pk_sc_processing_log"
    )
    op.execute(
        "ALTER TABLE sc_processing_log "
        "RENAME CONSTRAINT fk_processing_log_capture_event_id_capture_event "
        "TO fk_sc_processing_log_capture_event_id_sc_capture_event"
    )


def downgrade() -> None:
    if not _has_table("sc_capture_event"):
        return

    op.execute(
        "ALTER TABLE sc_processing_log "
        "RENAME CONSTRAINT fk_sc_processing_log_capture_event_id_sc_capture_event "
        "TO fk_processing_log_capture_event_id_capture_event"
    )
    op.execute(
        "ALTER TABLE sc_processing_log "
        "RENAME CONSTRAINT pk_sc_processing_log TO pk_processing_log"
    )
    op.execute(
        "ALTER TABLE sc_prediction "
        "RENAME CONSTRAINT fk_sc_prediction_capture_event_id_sc_capture_event "
        "TO fk_prediction_capture_event_id_capture_event"
    )
    op.execute(
        "ALTER TABLE sc_prediction "
        "RENAME CONSTRAINT pk_sc_prediction TO pk_prediction"
    )
    op.execute(
        "ALTER TABLE sc_capture_event "
        "RENAME CONSTRAINT uq_sc_capture_event_uuid TO uq_capture_event_uuid"
    )
    op.execute(
        "ALTER TABLE sc_capture_event "
        "RENAME CONSTRAINT pk_sc_capture_event TO pk_capture_event"
    )

    op.execute(
        "ALTER INDEX ix_sc_processing_log_capture_event_id "
        "RENAME TO ix_processing_log_capture_event_id"
    )
    op.execute(
        "ALTER INDEX ix_sc_prediction_capture_event_id "
        "RENAME TO ix_prediction_capture_event_id"
    )
    op.execute(
        "ALTER INDEX ix_sc_capture_event_status_created_at "
        "RENAME TO ix_capture_event_status_created_at"
    )
    op.execute("ALTER INDEX ix_sc_capture_event_status RENAME TO ix_capture_event_status")
    op.execute("ALTER INDEX ix_sc_capture_event_device_id RENAME TO ix_capture_event_device_id")

    op.rename_table("sc_processing_log", "processing_log")
    op.rename_table("sc_prediction", "prediction")
    op.rename_table("sc_capture_event", "capture_event")