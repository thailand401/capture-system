"""Vector index abstraction with a FAISS inner-product implementation.

Vectors are expected to be L2-normalized, so inner product == cosine
similarity. The index stores vectors in insertion order; positional ids
returned by :meth:`search` line up 1:1 with :class:`~ai.memory.vector_store.VectorStore`
entries, which is how a hit maps back to a sign id + image path.

``VectorIndex`` is an interface so tests can substitute a pure-numpy
brute-force index without installing FAISS.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai.utils.logging import get_logger

if TYPE_CHECKING:
    import numpy as np

logger = get_logger("memory.index")


class VectorIndex(ABC):
    """Minimal nearest-neighbour index interface used by the memory manager."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Vector dimensionality."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of vectors currently stored."""

    @abstractmethod
    def add(self, vectors: "np.ndarray") -> None:
        """Append a ``(n, dim)`` batch of vectors."""

    @abstractmethod
    def search(self, queries: "np.ndarray", k: int) -> tuple["np.ndarray", "np.ndarray"]:
        """Return ``(scores, ids)`` arrays, each shaped ``(n_queries, k)``."""

    @abstractmethod
    def reconstruct_all(self) -> "np.ndarray":
        """Return all stored vectors as a ``(size, dim)`` array (insertion order)."""

    @abstractmethod
    def reset(self) -> None:
        """Drop all stored vectors."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the index to ``path``."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load the index from ``path`` (replacing current contents)."""


class FaissIndex(VectorIndex):
    """FAISS ``IndexFlatIP`` wrapper (exact cosine search on unit vectors)."""

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._index: Any | None = None

    def _ensure_index(self) -> Any:
        if self._index is None:
            import faiss

            self._index = faiss.IndexFlatIP(self._dim)
        return self._index

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def size(self) -> int:
        return int(self._ensure_index().ntotal)

    def add(self, vectors: "np.ndarray") -> None:
        import numpy as np

        index = self._ensure_index()
        batch = np.ascontiguousarray(vectors, dtype=np.float32)
        if batch.ndim != 2 or batch.shape[1] != self._dim:
            raise ValueError(f"Expected (n, {self._dim}) vectors, got {batch.shape}")
        index.add(batch)
        logger.info("faiss_add", added=int(batch.shape[0]), total=self.size)

    def search(self, queries: "np.ndarray", k: int) -> tuple["np.ndarray", "np.ndarray"]:
        import numpy as np

        index = self._ensure_index()
        q = np.ascontiguousarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        k = min(k, max(index.ntotal, 1))
        scores, ids = index.search(q, k)
        return scores, ids

    def reconstruct_all(self) -> "np.ndarray":
        import numpy as np

        index = self._ensure_index()
        if index.ntotal == 0:
            return np.empty((0, self._dim), dtype=np.float32)
        return index.reconstruct_n(0, index.ntotal).astype(np.float32)

    def reset(self) -> None:
        import faiss

        self._index = faiss.IndexFlatIP(self._dim)

    def save(self, path: str | Path) -> None:
        import faiss

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._ensure_index(), str(path))
        logger.info("faiss_saved", path=str(path), total=self.size)

    def load(self, path: str | Path) -> None:
        import faiss

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"FAISS index not found: {path}")
        self._index = faiss.read_index(str(path))
        self._dim = int(self._index.d)
        logger.info("faiss_loaded", path=str(path), total=self.size)
