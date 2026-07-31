"""Per-event processing for the background worker.

Given a single claimed ``CaptureEvent`` (already transitioned to
``DOWNLOADING`` by the repository's atomic claim), this module downloads
its image from Supabase Storage, runs the AI pipeline, persists the
resulting prediction, deletes the temporary image from storage, and
transitions the event to ``DONE``. Every action is recorded in
``processing_log`` for observability. Each stage commits its own
transaction, so a failure part-way through never discards work already
safely persisted (e.g. a saved prediction survives even if the subsequent
storage cleanup fails).
"""

from __future__ import annotations

from pathlib import Path

from app.ai.interfaces import Pipeline
from app.ai.types import PipelineResult
from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import session_scope
from app.models.enums import CaptureStatus
from app.models.prediction import Prediction
from app.repositories.capture_event_repository import CaptureEventRepository
from app.repositories.prediction_repository import PredictionRepository
from app.repositories.processing_log_repository import ProcessingLogRepository
from app.storage.base import StorageBackend

logger = get_logger(__name__)


class EventProcessor:
    """Processes a single capture event from ``DOWNLOADING`` through to ``DONE``/``ERROR``."""

    def __init__(self, storage: StorageBackend, pipeline: Pipeline) -> None:
        self._storage = storage
        self._pipeline = pipeline
        self._settings = get_settings()

    async def process(self, event_id: int) -> None:
        """Run the full download -> AI -> persist -> cleanup lifecycle for one event."""
        local_image_path: Path | None = None
        try:
            local_image_path = await self._download_stage(event_id)
            await self._predict_stage(event_id, local_image_path)
            await self._cleanup_and_complete_stage(event_id)
        except Exception as exc:  # noqa: BLE001 - the worker loop must never crash
            logger.error("event_processing_failed", event_id=event_id, error=str(exc))
            await self._mark_error(event_id, str(exc))
        finally:
            if local_image_path is not None and local_image_path.exists():
                local_image_path.unlink(missing_ok=True)
                logger.info("local_cache_deleted", event_id=event_id, path=str(local_image_path))

    async def _download_stage(self, event_id: int) -> Path:
        """Download the event's image to the local cache directory."""
        async with session_scope() as session:
            capture_event_repo = CaptureEventRepository(session)
            processing_log_repo = ProcessingLogRepository(session)

            event = await capture_event_repo.get_by_id(event_id)
            if event is None:
                raise ValueError(f"Capture event {event_id} disappeared before download.")

            image_bytes = await self._storage.download(event.image_path)

            self._settings.local_cache.mkdir(parents=True, exist_ok=True)
            local_path = self._settings.local_cache / f"{event.uuid}.jpg"
            local_path.write_bytes(image_bytes)

            await processing_log_repo.add(
                action="download",
                message="Image downloaded from storage to local cache.",
                capture_event_id=event.id,
                detail={"image_path": event.image_path, "local_path": str(local_path)},
            )
            logger.info("image_downloaded", event_id=event_id, image_path=event.image_path)
            return local_path

    async def _predict_stage(self, event_id: int, local_image_path: Path) -> None:
        """Run the AI pipeline and persist its result as a new prediction row."""
        async with session_scope() as session:
            capture_event_repo = CaptureEventRepository(session)
            prediction_repo = PredictionRepository(session)
            processing_log_repo = ProcessingLogRepository(session)

            event = await capture_event_repo.get_by_id(event_id)
            if event is None:
                raise ValueError(f"Capture event {event_id} disappeared before processing.")

            await capture_event_repo.update_status(event, CaptureStatus.PROCESSING)

            result: PipelineResult = await self._pipeline.run(str(local_image_path))

            prediction = Prediction(
                capture_event_id=event.id,
                model_name=result.model_name,
                model_version=result.model_version,
                traffic_sign_class=result.traffic_sign_class,
                confidence=result.confidence,
                ocr_text=result.ocr_text,
                validation_score=result.validation_score,
            )
            await prediction_repo.create(prediction)

            await processing_log_repo.add(
                action="predict",
                message="AI pipeline finished; prediction saved.",
                capture_event_id=event.id,
                detail={"traffic_sign_class": result.traffic_sign_class, "confidence": result.confidence},
            )
            logger.info("prediction_saved", event_id=event_id, traffic_sign_class=result.traffic_sign_class)

    async def _cleanup_and_complete_stage(self, event_id: int) -> None:
        """Delete the temporary storage objects and mark the event DONE."""
        async with session_scope() as session:
            capture_event_repo = CaptureEventRepository(session)
            processing_log_repo = ProcessingLogRepository(session)

            event = await capture_event_repo.get_by_id(event_id)
            if event is None:
                raise ValueError(f"Capture event {event_id} disappeared before cleanup.")

            await self._storage.delete(event.image_path)
            await self._storage.delete(event.thumbnail_path)
            await processing_log_repo.add(
                action="delete",
                message="Temporary image and thumbnail deleted from storage.",
                capture_event_id=event.id,
                detail={"image_path": event.image_path, "thumbnail_path": event.thumbnail_path},
            )
            logger.info("storage_images_deleted", event_id=event_id)

            await capture_event_repo.update_status(event, CaptureStatus.DONE)
            await processing_log_repo.add(
                action="complete",
                message="Capture event processing completed successfully.",
                capture_event_id=event.id,
            )
            logger.info("event_done", event_id=event_id)

    async def _mark_error(self, event_id: int, error_message: str) -> None:
        """Best-effort transition of the event to ERROR; never raises."""
        try:
            async with session_scope() as session:
                capture_event_repo = CaptureEventRepository(session)
                processing_log_repo = ProcessingLogRepository(session)

                event = await capture_event_repo.get_by_id(event_id)
                if event is None:
                    return
                await capture_event_repo.update_status(event, CaptureStatus.ERROR)
                await processing_log_repo.add(
                    action="error",
                    level="ERROR",
                    message=error_message,
                    capture_event_id=event.id,
                )
        except Exception:  # noqa: BLE001 - error-handling itself must never crash the worker
            logger.error("event_mark_error_failed", event_id=event_id)
