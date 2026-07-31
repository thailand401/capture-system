"""Abstract interfaces for every AI pipeline stage.

Each stage is defined as an abstract base class so concrete
implementations (YOLO detector, PaddleOCR engine, etc.) can be swapped
independently without touching the pipeline orchestration or the worker.
Only stub implementations exist today — real model integrations are added
later behind these same interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.ai.types import (
    ClassificationResult,
    ColorValidationResult,
    DetectionResult,
    OcrResult,
    PipelineResult,
    ShapeValidationResult,
    ValidationResult,
)


class Detector(ABC):
    """Locates candidate traffic-sign regions in a full image (e.g. YOLO)."""

    @abstractmethod
    async def detect(self, image: np.ndarray) -> DetectionResult:
        """Return candidate bounding boxes for traffic signs in ``image``."""


class ShapeValidator(ABC):
    """Validates that a cropped region has a plausible traffic-sign shape."""

    @abstractmethod
    async def validate(self, image_crop: np.ndarray) -> ShapeValidationResult:
        """Check the geometric shape (circle/triangle/octagon/...) of a crop."""


class ColorValidator(ABC):
    """Validates that a cropped region has a plausible traffic-sign color palette."""

    @abstractmethod
    async def validate(self, image_crop: np.ndarray) -> ColorValidationResult:
        """Check the dominant colors of a crop against expected sign palettes."""


class OCR(ABC):
    """Extracts text printed on a sign (e.g. speed limit numbers)."""

    @abstractmethod
    async def read_text(self, image_crop: np.ndarray) -> OcrResult:
        """Run OCR on a cropped region and return any recognized text."""


class Classifier(ABC):
    """Classifies a validated crop into a specific traffic-sign class."""

    @abstractmethod
    async def classify(self, image_crop: np.ndarray) -> ClassificationResult:
        """Return the most likely traffic-sign class for a crop."""


class ValidationEngine(ABC):
    """Combines shape/color/OCR/classification signals into a final verdict."""

    @abstractmethod
    async def evaluate(
        self,
        *,
        shape_result: ShapeValidationResult,
        color_result: ColorValidationResult,
        ocr_result: OcrResult,
        classification_result: ClassificationResult,
    ) -> ValidationResult:
        """Aggregate individual stage results into one validation score."""


class Pipeline(ABC):
    """Full end-to-end AI pipeline run once per downloaded capture image."""

    @abstractmethod
    async def run(self, image_path: str) -> PipelineResult:
        """Run detection through validation on the image at ``image_path``."""
