"""Embedding value object produced by the embedding engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True, slots=True)
class Embedding:
    """A single L2-normalized feature vector plus the model that produced it.

    The vector is stored as a 1-D ``float32`` numpy array so it can be added
    directly to a FAISS index without further copying/casting.
    """

    vector: "np.ndarray"
    model_name: str

    @property
    def dim(self) -> int:
        return int(self.vector.shape[-1])
