"""Traffic Sign Recognition Engine.

A dataset-driven image *retrieval* engine (not a fixed-class classifier):

    YOLO detection -> crop -> DINOv2 embedding -> FAISS memory search
    -> voting -> shape/color/blur validation -> rule engine -> Prediction

The Traffic Sign Memory (FAISS index + metadata) is the source of truth.
Adding new sample images only appends vectors — no retraining is required
and the system becomes more accurate as the memory grows.

Every stage is defined behind an interface and wired with dependency
injection (see :mod:`ai.factory`), so any component can be replaced
independently.
"""

from __future__ import annotations

from ai.config import EngineConfig, RuleWeights
from ai.models.bbox import BoundingBox
from ai.models.embedding import Embedding
from ai.models.prediction import Prediction, TopKMatch

__all__ = [
    "BoundingBox",
    "Embedding",
    "EngineConfig",
    "Prediction",
    "RuleWeights",
    "TopKMatch",
    "build_pipeline",
]


def build_pipeline(config: "EngineConfig | None" = None):
    """Convenience wrapper around :func:`ai.factory.build_pipeline`.

    Imported lazily so ``import ai`` stays cheap and does not pull in the
    heavy CV/ML stack (torch, faiss, ultralytics) unless a real pipeline is
    actually constructed.
    """
    from ai.factory import build_pipeline as _build

    return _build(config)
