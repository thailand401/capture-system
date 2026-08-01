"""Blur validation via the variance of the Laplacian.

A low Laplacian variance means few sharp edges, i.e. a blurry crop. The
score is normalized to ``[0, 1]`` where 1.0 is sharp and 0.0 is heavily
blurred, using a configurable variance threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ai.utils.logging import get_logger

if TYPE_CHECKING:
    import numpy as np

logger = get_logger("validator.blur")


@dataclass(frozen=True, slots=True)
class BlurResult:
    """Sharpness score (0 blurry .. 1 sharp) plus the raw Laplacian variance."""

    score: float
    variance: float
    is_blurry: bool


class BlurValidator:
    """Score crop sharpness using the variance of the Laplacian."""

    def __init__(self, *, threshold: float = 150.0) -> None:
        self._threshold = threshold

    def validate(self, crop: "np.ndarray") -> BlurResult:
        """Return a sharpness score for ``crop``."""
        import cv2

        if crop.size == 0:
            return BlurResult(score=0.0, variance=0.0, is_blurry=True)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        score = max(0.0, min(1.0, variance / self._threshold))
        is_blurry = variance < self._threshold

        logger.info("blur_validated", variance=round(variance, 2), score=round(score, 4), blurry=is_blurry)
        return BlurResult(score=score, variance=variance, is_blurry=is_blurry)
