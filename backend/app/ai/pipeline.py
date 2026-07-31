"""Default AI pipeline orchestrator.

Wires together a ``Detector``, ``ShapeValidator``, ``ColorValidator``,
``OCR`` engine, ``Classifier`` and ``ValidationEngine`` into a single
``Pipeline``. Every dependency is injected via the constructor, so any
stage can be swapped (e.g. stub -> real YOLO) without changing this
orchestration logic.
"""

from __future__ import annotations

import cv2

from app.ai.interfaces import (
    OCR,
    Classifier,
    ColorValidator,
    Detector,
    Pipeline,
    ShapeValidator,
    ValidationEngine,
)
from app.ai.types import PipelineResult
from app.core.logging import get_logger

logger = get_logger(__name__)


class DefaultPipeline(Pipeline):
    """Runs detection, validation, OCR and classification against one image."""

    def __init__(
        self,
        *,
        detector: Detector,
        shape_validator: ShapeValidator,
        color_validator: ColorValidator,
        ocr: OCR,
        classifier: Classifier,
        validation_engine: ValidationEngine,
        model_name: str = "yolo",
        model_version: str = "stub",
    ) -> None:
        self._detector = detector
        self._shape_validator = shape_validator
        self._color_validator = color_validator
        self._ocr = ocr
        self._classifier = classifier
        self._validation_engine = validation_engine
        self._model_name = model_name
        self._model_version = model_version

    async def run(self, image_path: str) -> PipelineResult:
        """Load the image at ``image_path`` and run it through every stage."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at '{image_path}'.")

        detection = await self._detector.detect(image)

        # Use the highest-confidence detection (if any) as the crop to
        # validate/classify. Falls back to the full image when nothing was
        # detected, so the pipeline can still run end-to-end with stubs.
        crop = image
        if detection.boxes:
            box = detection.boxes[0]
            crop = image[box.y : box.y + box.height, box.x : box.x + box.width]

        shape_result = await self._shape_validator.validate(crop)
        color_result = await self._color_validator.validate(crop)
        ocr_result = await self._ocr.read_text(crop)
        classification_result = await self._classifier.classify(crop)

        validation = await self._validation_engine.evaluate(
            shape_result=shape_result,
            color_result=color_result,
            ocr_result=ocr_result,
            classification_result=classification_result,
        )

        logger.info(
            "pipeline_run_completed",
            image_path=image_path,
            traffic_sign_class=classification_result.traffic_sign_class,
            validation_score=validation.validation_score,
        )

        return PipelineResult(
            model_name=self._model_name,
            model_version=self._model_version,
            traffic_sign_class=classification_result.traffic_sign_class,
            confidence=classification_result.confidence,
            ocr_text=ocr_result.text,
            validation_score=validation.validation_score,
        )
