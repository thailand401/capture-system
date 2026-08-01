"""MemoryManager: the source of truth for traffic sign recognition.

Owns the FAISS index + metadata store and keeps them consistent. Building
the memory scans ``dataset/<sign_id>/*.png`` folders, embeds every image and
persists the result. Recognition accuracy improves simply by adding more
sample images — :meth:`append` adds vectors incrementally with **no
retraining and no full rebuild**.

Public API (as specified): ``build``, ``load``, ``save``, ``append``,
``search``, ``rebuild``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ai.embedding.embedding_engine import EmbeddingEngine
from ai.memory.faiss_index import VectorIndex
from ai.memory.vector_store import VectorStore
from ai.models.embedding import Embedding
from ai.utils.image_io import load_image
from ai.utils.logging import get_logger

if TYPE_CHECKING:
    import numpy as np

logger = get_logger("memory.manager")


@dataclass(frozen=True, slots=True)
class MemoryMatch:
    """A single nearest-neighbour hit from the memory."""

    sign_id: str
    image_path: str
    similarity: float
    vector_id: int


class MemoryManager:
    """Coordinates the embedding engine, vector index and metadata store."""

    def __init__(
        self,
        *,
        embedding_engine: EmbeddingEngine,
        index: VectorIndex,
        store: VectorStore,
        dataset_dir: str | Path = "dataset",
        index_path: str | Path = "memory/vectors.faiss",
        metadata_path: str | Path = "memory/metadata.json",
        image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp"),
    ) -> None:
        self._engine = embedding_engine
        self._index = index
        self._store = store
        self._dataset_dir = Path(dataset_dir)
        self._index_path = Path(index_path)
        self._metadata_path = Path(metadata_path)
        self._extensions = tuple(ext.lower() for ext in image_extensions)

    @property
    def embedding_engine(self) -> EmbeddingEngine:
        """The engine used to embed both dataset and query images."""
        return self._engine

    # --- Lifecycle -------------------------------------------------------

    def exists(self) -> bool:
        """True when both persisted memory files are present."""
        return self._index_path.exists() and self._metadata_path.exists()

    def ensure_ready(self) -> None:
        """Load persisted memory if present, otherwise build it from dataset."""
        if self.exists():
            self.load()
        else:
            self.build()

    def build(self) -> int:
        """Scan the dataset, embed every image and persist a fresh memory.

        Returns the number of vectors stored.
        """
        pairs = self._scan_dataset()
        if not pairs:
            raise FileNotFoundError(f"No dataset images found under {self._dataset_dir}")

        self._index.reset()
        self._store.reset()

        images: list[np.ndarray] = []
        for image_path, _sign_id in pairs:
            images.append(load_image(image_path))

        vectors = self._engine.encode_batch(images)
        self._index.add(vectors)
        for image_path, sign_id in pairs:
            self._store.add(sign_id=sign_id, image_path=str(image_path))

        self.save()
        logger.info("memory_built", vectors=self._store.size, signs=len({s for _, s in pairs}))
        return self._store.size

    def rebuild(self) -> int:
        """Alias for a full rebuild from the dataset (clears existing memory)."""
        logger.info("memory_rebuild_requested")
        return self.build()

    def load(self) -> None:
        """Load the persisted FAISS index and metadata."""
        self._index.load(self._index_path)
        self._store.load(self._metadata_path)
        if self._index.size != self._store.size:
            raise ValueError(
                f"Corrupt memory: index has {self._index.size} vectors "
                f"but metadata has {self._store.size} entries"
            )
        logger.info("memory_loaded", vectors=self._store.size)

    def save(self) -> None:
        """Persist the FAISS index and metadata to disk."""
        self._index.save(self._index_path)
        self._store.save(self._metadata_path)

    # --- Incremental updates --------------------------------------------

    def append(self, image_path: str | Path, sign_id: str, *, persist: bool = True) -> MemoryMatch:
        """Embed a single new sample image and append it to the memory.

        This is the mechanism for improving accuracy over time: no rebuild,
        only a vector append + metadata entry.
        """
        image = load_image(image_path)
        vector = self._engine.encode_batch([image])
        self._index.add(vector)
        entry = self._store.add(sign_id=sign_id, image_path=str(image_path))
        if persist:
            self.save()
        logger.info("memory_appended", sign_id=sign_id, image_path=str(image_path), total=self._store.size)
        return MemoryMatch(sign_id=sign_id, image_path=str(image_path), similarity=1.0, vector_id=entry.vector_id)

    def append_directory(self, directory: str | Path, sign_id: str, *, persist: bool = True) -> int:
        """Append every image in ``directory`` under a single ``sign_id``."""
        directory = Path(directory)
        added = 0
        for image_path in sorted(directory.iterdir()):
            if image_path.suffix.lower() in self._extensions:
                self.append(image_path, sign_id, persist=False)
                added += 1
        if persist and added:
            self.save()
        return added

    def append_vector(
        self, vector: "np.ndarray", sign_id: str, image_path: str, *, persist: bool = True
    ) -> MemoryMatch:
        """Append a *precomputed* embedding (e.g. a promoted candidate vector).

        Unlike :meth:`append`, no image is re-read or re-encoded — the caller
        already holds the normalized vector.
        """
        import numpy as np

        batch = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        self._index.add(batch)
        entry = self._store.add(sign_id=sign_id, image_path=str(image_path))
        if persist:
            self.save()
        logger.info("memory_vector_appended", sign_id=sign_id, total=self._store.size)
        return MemoryMatch(sign_id=sign_id, image_path=str(image_path), similarity=1.0, vector_id=entry.vector_id)

    def reconstruct_all(self) -> "np.ndarray":
        """Return every stored permanent vector aligned with ``store.entries``."""
        return self._index.reconstruct_all()

    def replace_all(self, vectors: "np.ndarray", entries: list[tuple[str, str]], *, persist: bool = True) -> int:
        """Reset the memory and repopulate it from ``vectors`` + ``(sign_id, path)``.

        Used by the Memory Optimizer to write back a pruned set of vectors.
        """
        import numpy as np

        batch = np.asarray(vectors, dtype=np.float32).reshape(-1, self._store.dim) if len(entries) else None
        self._index.reset()
        self._store.reset()
        if batch is not None and len(entries):
            self._index.add(batch)
            for sign_id, image_path in entries:
                self._store.add(sign_id=sign_id, image_path=image_path)
        if persist:
            self.save()
        logger.info("memory_replaced", vectors=self._store.size)
        return self._store.size

    @property
    def store(self) -> VectorStore:
        """The underlying metadata store (read-only introspection)."""
        return self._store

    # --- Query -----------------------------------------------------------

    def search(self, embedding: Embedding, k: int = 20) -> list[MemoryMatch]:
        """Return the top-``k`` nearest memory entries for ``embedding``."""
        if self._store.size == 0:
            return []
        scores, ids = self._index.search(embedding.vector.reshape(1, -1), k)
        matches: list[MemoryMatch] = []
        for score, vector_id in zip(scores[0], ids[0], strict=False):
            if vector_id < 0:
                continue
            entry = self._store.get(int(vector_id))
            matches.append(
                MemoryMatch(
                    sign_id=entry.sign_id,
                    image_path=entry.image_path,
                    similarity=float(score),
                    vector_id=int(vector_id),
                )
            )
        return matches

    # --- Internal --------------------------------------------------------

    def _scan_dataset(self) -> list[tuple[Path, str]]:
        """Return ``(image_path, sign_id)`` pairs for every dataset image.

        The immediate subdirectory name *is* the official sign id — nothing
        is hardcoded; the class set is whatever exists on disk.
        """
        pairs: list[tuple[Path, str]] = []
        if not self._dataset_dir.exists():
            return pairs
        for sign_dir in sorted(p for p in self._dataset_dir.iterdir() if p.is_dir()):
            sign_id = sign_dir.name
            for image_path in sorted(sign_dir.iterdir()):
                if image_path.suffix.lower() in self._extensions:
                    pairs.append((image_path, sign_id))
        return pairs
