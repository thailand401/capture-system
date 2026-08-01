"""Color validation in HSV space.

Reports the dominant traffic-sign-relevant colors present in a crop and a
score reflecting how much of the crop is covered by sign-typical colors
(red / blue / yellow / green / white / black).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ai.utils.logging import get_logger

if TYPE_CHECKING:
    import numpy as np

logger = get_logger("validator.color")

# HSV ranges (OpenCV: H in 0..179, S/V in 0..255). Red wraps the hue circle.
_HSV_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red": [((0, 80, 50), (10, 255, 255)), ((170, 80, 50), (179, 255, 255))],
    "yellow": [((20, 80, 80), (35, 255, 255))],
    "green": [((40, 60, 40), (85, 255, 255))],
    "blue": [((90, 60, 40), (130, 255, 255))],
    "white": [((0, 0, 190), (179, 40, 255))],
    "black": [((0, 0, 0), (179, 255, 60))],
}


@dataclass(frozen=True, slots=True)
class ColorResult:
    """Dominant colors (fraction >= threshold) plus a coverage score."""

    dominant_colors: list[str] = field(default_factory=list)
    score: float = 0.0
    fractions: dict[str, float] = field(default_factory=dict)


class ColorValidator:
    """Detect dominant sign colors and score color-plausibility."""

    def __init__(self, *, dominant_threshold: float = 0.12) -> None:
        self._dominant_threshold = dominant_threshold

    def validate(self, crop: "np.ndarray") -> ColorResult:
        """Return dominant colors and a coverage score for ``crop``."""
        import cv2
        import numpy as np

        if crop.size == 0 or crop.ndim != 3:
            return ColorResult()

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        total = float(crop.shape[0] * crop.shape[1])

        fractions: dict[str, float] = {}
        for color, ranges in _HSV_RANGES.items():
            mask = None
            for lower, upper in ranges:
                band = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
                mask = band if mask is None else cv2.bitwise_or(mask, band)
            fractions[color] = float(cv2.countNonZero(mask)) / total if mask is not None else 0.0

        dominant = sorted(
            (c for c, f in fractions.items() if f >= self._dominant_threshold),
            key=lambda c: fractions[c],
            reverse=True,
        )
        # Score = coverage by chromatic sign colors (exclude generic black bg).
        score = min(1.0, sum(fractions[c] for c in ("red", "blue", "yellow", "green", "white")))

        logger.info("color_validated", dominant=dominant, score=round(score, 4))
        return ColorResult(dominant_colors=dominant, score=score, fractions=fractions)
