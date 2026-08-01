"""Image IO helpers (thin wrappers over OpenCV, lazily imported)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def image_hash(image: "np.ndarray") -> str:
    """Return a stable content hash (sha256 hex) of an image's raw pixels."""
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def load_image(image_path: str | Path) -> "np.ndarray":
    """Read an image from disk as a BGR ``numpy`` array.

    Raises:
        FileNotFoundError: The path does not exist.
        ValueError: The file exists but could not be decoded as an image.
    """
    import cv2

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def save_image(image: "np.ndarray", image_path: str | Path) -> Path:
    """Write a BGR ``numpy`` image to disk, creating parent dirs as needed."""
    import cv2

    path = Path(image_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise ValueError(f"Could not write image: {path}")
    return path
