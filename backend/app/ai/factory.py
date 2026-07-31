"""Factory for building the configured AI ``Pipeline`` instance.

Centralizing construction here means the worker only depends on the
``Pipeline`` interface and this factory — never on concrete stage
implementations directly.
"""

from __future__ import annotations

from app.ai.interfaces import Pipeline
from app.ai.pipeline import DefaultPipeline
from app.ai.stubs import (
    StubClassifier,
    StubColorValidator,
    StubDetector,
    StubOCR,
    StubShapeValidator,
    StubValidationEngine,
)
from app.core.config import get_settings


def build_default_pipeline() -> Pipeline:
    """Construct the pipeline currently configured for the application.

    Today this always wires the stub implementations together. Once real
    models are ready, swap the concrete classes passed in here (e.g. a
    ``YoloDetector`` reading ``settings.yolo_model``) — no other code needs
    to change.
    """
    settings = get_settings()
    return DefaultPipeline(
        detector=StubDetector(),
        shape_validator=StubShapeValidator(),
        color_validator=StubColorValidator(),
        ocr=StubOCR(),
        classifier=StubClassifier(),
        validation_engine=StubValidationEngine(),
        model_name="yolo",
        model_version=settings.yolo_model,
    )
