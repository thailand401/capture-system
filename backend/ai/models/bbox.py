"""Pixel-space bounding box value object."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """An axis-aligned bounding box in ``xyxy`` pixel coordinates.

    ``(x1, y1)`` is the top-left corner and ``(x2, y2)`` the bottom-right
    corner. Coordinates are floats so sub-pixel model outputs are preserved
    until an explicit :meth:`to_int` conversion is requested.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"Invalid bounding box: {self!r}")

    @classmethod
    def from_xywh(cls, x: float, y: float, width: float, height: float) -> "BoundingBox":
        """Build a box from top-left origin plus width/height."""
        return cls(x1=x, y1=y, x2=x + width, y2=y + height)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_int(self) -> tuple[int, int, int, int]:
        """Return integer ``(x1, y1, x2, y2)`` suitable for array slicing."""
        return (int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2)))

    def padded(self, fraction: float, image_width: int, image_height: int) -> "BoundingBox":
        """Return a copy grown by ``fraction`` on each side, clamped to image bounds."""
        pad_x = self.width * fraction
        pad_y = self.height * fraction
        return BoundingBox(
            x1=max(0.0, self.x1 - pad_x),
            y1=max(0.0, self.y1 - pad_y),
            x2=min(float(image_width), self.x2 + pad_x),
            y2=min(float(image_height), self.y2 + pad_y),
        )
