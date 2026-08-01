"""L2 feature normalization.

Cosine similarity via FAISS inner-product requires unit-length vectors.
Centralizing normalization here guarantees the memory-build path and the
query path normalize identically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class FeatureNormalizer:
    """Row-wise L2 normalizer for embedding matrices/vectors."""

    def __init__(self, *, epsilon: float = 1e-12) -> None:
        self._epsilon = epsilon

    def normalize(self, vectors: "np.ndarray") -> "np.ndarray":
        """Return ``vectors`` L2-normalized along the last axis as ``float32``.

        Accepts a single vector (shape ``(d,)``) or a batch (shape ``(n, d)``).
        Zero vectors are left as zeros rather than producing NaNs.
        """
        import numpy as np

        array = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(array, axis=-1, keepdims=True)
        norms = np.maximum(norms, self._epsilon)
        return (array / norms).astype(np.float32)
