"""Stub AI pipeline implementations.

These placeholders satisfy the pipeline interfaces so the rest of the
system (worker, repositories, API) can be built, tested, and deployed
before real computer-vision models are integrated. Each stub is a drop-in
replacement point: swap it out for a real implementation without touching
any other module.
"""

from __future__ import annotations

import numpy as np

from app.ai.interfaces import OCR, Classifier, ColorValidator, Detector, ShapeValidator, ValidationEngine
from app.ai.types import (
    BoundingBox,
    ClassificationResult,
    ColorValidationResult,
    DetectionResult,
    OcrResult,
    ShapeValidationResult,
    ValidationResult,
)


class StubDetector(Detector):
    """Placeholder detector. TODO: replace with an Ultralytics YOLO model."""

    async def detect(self, image: np.ndarray) -> DetectionResult:
        height, width = image.shape[:2]
        # Return a single full-frame "detection" so downstream stages have
        # something to operate on until real detection is implemented.
        return DetectionResult(
            boxes=[BoundingBox(x=0, y=0, width=width, height=height)],
            scores=[0.0],
            raw_class_ids=[-1],
        )


class StubShapeValidator(ShapeValidator):
    """Placeholder shape validator. TODO: replace with real contour analysis."""

    async def validate(self, image_crop: np.ndarray) -> ShapeValidationResult:
        return ShapeValidationResult(is_valid=False, shape=None, score=0.0)


class StubColorValidator(ColorValidator):
    """Placeholder color validator. TODO: replace with HSV palette matching."""

    async def validate(self, image_crop: np.ndarray) -> ColorValidationResult:
        return ColorValidationResult(is_valid=False, dominant_color=None, score=0.0)


class StubOCR(OCR):
    """Placeholder OCR engine. TODO: replace with PaddleOCR."""

    async def read_text(self, image_crop: np.ndarray) -> OcrResult:
        return OcrResult(text=None, confidence=0.0)


class StubClassifier(Classifier):
    """Placeholder classifier. TODO: replace with a trained sign classifier."""

    async def classify(self, image_crop: np.ndarray) -> ClassificationResult:
        return ClassificationResult(traffic_sign_class=None, confidence=0.0)


class StubValidationEngine(ValidationEngine):
    """Placeholder validation engine combining stub stage results."""

    async def evaluate(
        self,
        *,
        shape_result: ShapeValidationResult,
        color_result: ColorValidationResult,
        ocr_result: OcrResult,
        classification_result: ClassificationResult,
    ) -> ValidationResult:
        return ValidationResult(
            is_valid=False, validation_score=0.0, reasons=["stub_pipeline_not_implemented"]
        )
