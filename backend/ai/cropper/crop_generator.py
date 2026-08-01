"""Crop generation from detections.

Produces exactly one crop image per detection and (optionally) persists it
to disk so the resulting :class:`~ai.models.prediction.Prediction` can point
at the exact pixels that were classified.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ai.models.bbox import BoundingBox
from ai.utils.image_io import save_image
from ai.utils.logging import get_logger

if TYPE_CHECKING:
    import numpy as np

logger = get_logger("cropper")


@dataclass(frozen=True, slots=True)
class Crop:
    """A cropped region plus where (if anywhere) it was written to disk."""

    image: "np.ndarray"
    bbox: BoundingBox
    path: str | None = None


class CropGenerator:
    """Generates crop images from bounding boxes, with optional padding + save."""

    def __init__(self, *, output_dir: str | Path = "crops", padding: float = 0.05) -> None:
        self._output_dir = Path(output_dir)
        self._padding = padding

    def generate(self, image: "np.ndarray", bbox: BoundingBox, *, save: bool = True) -> Crop:
        """Crop ``bbox`` out of ``image`` (with padding) and optionally save it."""
        height, width = image.shape[:2]
        box = bbox.padded(self._padding, image_width=width, image_height=height) if self._padding else bbox
        x1, y1, x2, y2 = box.to_int()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Degenerate crop region for bbox={bbox!r} on {width}x{height} image")

        crop_image = image[y1:y2, x1:x2].copy()

        path: str | None = None
        if save:
            out_path = self._output_dir / f"crop_{uuid.uuid4().hex}.png"
            path = str(save_image(crop_image, out_path))

        logger.info("crop_generated", bbox=box.to_int(), path=path)
        return Crop(image=crop_image, bbox=box, path=path)
