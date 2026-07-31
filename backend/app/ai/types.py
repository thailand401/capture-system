"""Shared value objects passed between AI pipeline stages.

These are plain dataclasses (not Pydantic models) because they are
internal domain objects used only in-process between pipeline stages —
they are never serialized directly over the wire.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """A pixel-space axis-aligned bounding box, ``x``/``y`` at top-left."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Output of a ``Detector``: candidate traffic-sign regions in an image."""

    boxes: list[BoundingBox] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    raw_class_ids: list[int] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ShapeValidationResult:
    """Output of a ``ShapeValidator``: does the region look like a sign shape?"""

    is_valid: bool
    shape: str | None = None
    score: float = 0.0


@dataclass(frozen=True, slots=True)
class ColorValidationResult:
    """Output of a ``ColorValidator``: does the region's color palette match?"""

    is_valid: bool
    dominant_color: str | None = None
    score: float = 0.0


@dataclass(frozen=True, slots=True)
class OcrResult:
    """Output of an ``OCR`` engine run against a cropped region."""

    text: str | None = None
    confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Output of a ``Classifier``: the specific traffic-sign class."""

    traffic_sign_class: str | None = None
    confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Output of a ``ValidationEngine``: combined confidence across stages."""

    is_valid: bool
    validation_score: float = 0.0
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Final result of running the full AI ``Pipeline`` against one image."""

    model_name: str
    model_version: str
    traffic_sign_class: str | None
    confidence: float | None
    ocr_text: str | None
    validation_score: float | None
