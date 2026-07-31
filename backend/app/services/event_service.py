"""Business logic for ingesting new capture events.

Keeps upload orchestration (storage + persistence + audit logging) out of
the API layer, per the project's clean-architecture requirement that
routes contain no business logic.
"""

from __future__ import annotations

from uuid import uuid4

from app.core.logging import get_logger
from app.models.capture_event import CaptureEvent
from app.repositories.capture_event_repository import CaptureEventRepository
from app.repositories.processing_log_repository import ProcessingLogRepository
from app.schemas.capture_event import EventMetadata
from app.storage.base import StorageBackend

logger = get_logger(__name__)


class EventService:
    """Orchestrates the ``POST /events`` use case: upload + persist + audit."""

    def __init__(
        self,
        storage: StorageBackend,
        capture_event_repo: CaptureEventRepository,
        processing_log_repo: ProcessingLogRepository,
    ) -> None:
        self._storage = storage
        self._capture_events = capture_event_repo
        self._processing_logs = processing_log_repo

    async def create_event(
        self,
        *,
        metadata: EventMetadata,
        image_bytes: bytes,
        image_content_type: str,
        thumbnail_bytes: bytes,
        thumbnail_content_type: str,
    ) -> CaptureEvent:
        """Upload the image/thumbnail to storage and persist a new NEW event.

        If anything fails after an image has already been uploaded (e.g.
        the thumbnail upload or the DB insert fails), any objects already
        uploaded for this event are deleted as a compensating action so no
        orphaned files are left behind in the bucket.
        """
        event_uuid = metadata.uuid or uuid4()
        image_path = f"{metadata.device_id}/{event_uuid}/image.jpg"
        thumbnail_path = f"{metadata.device_id}/{event_uuid}/thumbnail.jpg"

        uploaded_paths: list[str] = []
        try:
            await self._storage.upload(image_path, image_bytes, image_content_type)
            uploaded_paths.append(image_path)
            await self._storage.upload(thumbnail_path, thumbnail_bytes, thumbnail_content_type)
            uploaded_paths.append(thumbnail_path)

            event = CaptureEvent(
                uuid=event_uuid,
                device_id=metadata.device_id,
                capture_time=metadata.capture_time,
                latitude=metadata.latitude,
                longitude=metadata.longitude,
                heading=metadata.heading,
                speed=metadata.speed,
                image_path=image_path,
                thumbnail_path=thumbnail_path,
            )
            await self._capture_events.create(event)

            await self._processing_logs.add(
                action="upload",
                message="Capture event uploaded from device.",
                capture_event_id=event.id,
                detail={"device_id": metadata.device_id, "image_path": image_path},
            )
        except Exception:
            logger.error("event_upload_failed", device_id=metadata.device_id, uploaded_paths=uploaded_paths)
            for path in uploaded_paths:
                try:
                    await self._storage.delete(path)
                except Exception:
                    logger.error("event_upload_compensation_delete_failed", path=path)
            raise

        logger.info("event_uploaded", event_uuid=str(event_uuid), device_id=metadata.device_id)
        return event
