"""Ultralytics YOLO detector, collapsed to a single ``traffic_sign`` class.

Whatever classes the underlying weights were trained on, every surviving
box is reported as ``traffic_sign``. The engine never trusts YOLO for the
*identity* of a sign — only its *location* and a detection confidence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ai.detector.detector import Detection, Detector
from ai.models.bbox import BoundingBox
from ai.utils.device import resolve_device
from ai.utils.logging import get_logger

if TYPE_CHECKING:
    import numpy as np

logger = get_logger("detector.yolo")


class YoloTrafficSignDetector(Detector):
    """Detector backed by an Ultralytics YOLO model.

    The model is loaded lazily on first use so constructing the detector is
    cheap and import-safe in environments without the CV stack installed.
    """

    def __init__(
        self,
        *,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.25,
        iou: float = 0.45,
        device: str | None = None,
        label: str = "traffic_sign",
    ) -> None:
        self._model_path = model_path
        self._confidence = confidence
        self._iou = iou
        self._device = device
        self._label = label
        self._model: Any | None = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            self._device = resolve_device(self._device)
            logger.info("loading_yolo_model", model=self._model_path, device=self._device)
            self._model = YOLO(self._model_path)
        return self._model

    def detect(self, image: "np.ndarray") -> list[Detection]:
        """Run YOLO on ``image`` and return single-class detections."""
        model = self._ensure_model()
        results = model.predict(
            source=image,
            conf=self._confidence,
            iou=self._iou,
            device=self._device,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                confidence = float(box.conf[0].item())
                detections.append(
                    Detection(
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        confidence=confidence,
                        label=self._label,
                    )
                )

        logger.info("yolo_detected", count=len(detections))
        return detections
