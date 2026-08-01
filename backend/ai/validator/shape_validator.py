"""Shape validation via contour analysis.

Classifies a crop's dominant outline as circle / triangle / rectangle /
square / octagon and returns a confidence-like score. Traffic signs have
strong geometric priors, so this is a cheap, model-free sanity signal for
the rule engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ai.utils.logging import get_logger

if TYPE_CHECKING:
    import numpy as np

logger = get_logger("validator.shape")


@dataclass(frozen=True, slots=True)
class ShapeResult:
    """Detected outline shape plus a 0..1 confidence score."""

    shape: str | None
    score: float
    vertices: int = 0


class ShapeValidator:
    """Detect the dominant geometric shape of a crop."""

    def __init__(self, *, approx_epsilon: float = 0.03) -> None:
        self._approx_epsilon = approx_epsilon

    def validate(self, crop: "np.ndarray") -> ShapeResult:
        """Return the outline shape and score for ``crop``."""
        import cv2
        import numpy as np

        if crop.size == 0:
            return ShapeResult(shape=None, score=0.0)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return ShapeResult(shape=None, score=0.0)

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0 or area <= 0:
            return ShapeResult(shape=None, score=0.0)

        approx = cv2.approxPolyDP(contour, self._approx_epsilon * perimeter, True)
        vertices = len(approx)
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        shape, score = self._classify(vertices, circularity, approx, solidity)
        logger.info("shape_validated", shape=shape, vertices=vertices, score=round(score, 4))
        return ShapeResult(shape=shape, score=score, vertices=vertices)

    @staticmethod
    def _classify(vertices: int, circularity: float, approx, solidity: float) -> tuple[str | None, float]:
        import cv2

        if vertices == 3:
            return "triangle", min(1.0, solidity)
        if vertices == 4:
            _, (w, h), _ = cv2.minAreaRect(approx)
            aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0.0
            shape = "square" if aspect >= 0.9 else "rectangle"
            return shape, min(1.0, solidity)
        if vertices == 8:
            return "octagon", min(1.0, solidity)
        if circularity >= 0.7:
            return "circle", min(1.0, circularity)
        return None, max(0.0, min(1.0, circularity))
