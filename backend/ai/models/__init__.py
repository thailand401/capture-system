"""Domain models (value objects) shared across engine stages."""

from __future__ import annotations

from ai.models.bbox import BoundingBox
from ai.models.embedding import Embedding
from ai.models.prediction import Prediction, TopKMatch

__all__ = ["BoundingBox", "Embedding", "Prediction", "TopKMatch"]
