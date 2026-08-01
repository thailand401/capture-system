"""Detector interface and its output value object.

The detector answers a single question: *where are the traffic signs?* It
deliberately emits only one class (``traffic_sign``) — the specific sign id
is decided later by embedding search, never by the detector. This keeps the
detector fully decoupled from the (open, dataset-driven) set of sign ids.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ai.models.bbox import BoundingBox

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True, slots=True)
class Detection:
    """A single detected region: where it is and how confident the detector is."""

    bbox: BoundingBox
    confidence: float
    label: str = "traffic_sign"


class Detector(ABC):
    """Locates candidate traffic-sign regions in a full BGR image."""

    @abstractmethod
    def detect(self, image: "np.ndarray") -> list[Detection]:
        """Return zero or more :class:`Detection` objects for ``image``."""
