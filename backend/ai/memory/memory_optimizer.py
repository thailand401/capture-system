"""Memory Optimizer: prune redundant vectors from Permanent Memory.

Within each traffic sign id, embeddings that are near-duplicates carry
little extra information. The optimizer greedily clusters same-sign vectors
by cosine similarity and keeps one representative per cluster (the first
seen), dropping the rest. Vectors are assumed L2-normalized.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai.utils.logging import get_logger

if TYPE_CHECKING:
    import numpy as np

logger = get_logger("memory.optimizer")


class MemoryOptimizer:
    """Greedy cosine-similarity de-duplication, scoped per sign id."""

    def __init__(self, *, redundancy_threshold: float = 0.98) -> None:
        self._threshold = redundancy_threshold

    def select_representatives(self, vectors: "np.ndarray", sign_ids: list[str]) -> list[int]:
        """Return indices of the vectors to keep (representatives).

        Two vectors of the *same* sign whose cosine similarity is
        >= ``redundancy_threshold`` are treated as redundant; only the
        earlier one is kept.
        """
        import numpy as np

        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(sign_ids):
            raise ValueError("vectors and sign_ids length mismatch")

        kept: list[int] = []
        kept_by_sign: dict[str, list[int]] = {}
        for i, sign_id in enumerate(sign_ids):
            reps = kept_by_sign.get(sign_id, [])
            is_redundant = any(
                float(np.dot(matrix[i], matrix[j])) >= self._threshold for j in reps
            )
            if is_redundant:
                continue
            kept.append(i)
            kept_by_sign.setdefault(sign_id, []).append(i)

        logger.info("prune_selected", kept=len(kept), total=len(sign_ids))
        return kept
