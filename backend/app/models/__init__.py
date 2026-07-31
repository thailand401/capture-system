"""ORM models package.

Import all model modules here so that ``Base.metadata`` is fully populated
for Alembic autogeneration and for ``Base.metadata.create_all`` in tests.
"""

from app.models.capture_event import CaptureEvent
from app.models.enums import CaptureStatus
from app.models.prediction import Prediction
from app.models.processing_log import ProcessingLog

__all__ = ["CaptureEvent", "CaptureStatus", "Prediction", "ProcessingLog"]
