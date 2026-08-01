"""Metadata store kept in lockstep with the vector index.

``metadata.json`` records, for every vector position, which traffic sign id
and source image it came from, plus the embedding model and creation time.
Entry order matches insertion order in the FAISS index, so a search hit at
position ``i`` maps to ``entries[i]``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from ai.utils.logging import get_logger

logger = get_logger("memory.store")


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """One stored vector's provenance."""

    vector_id: int
    sign_id: str
    image_path: str
    embedding_model: str
    created_time: str


class VectorStore:
    """In-memory list of :class:`MemoryEntry` with JSON persistence."""

    def __init__(self, *, embedding_model: str, dim: int) -> None:
        self._embedding_model = embedding_model
        self._dim = dim
        self._entries: list[MemoryEntry] = []

    @property
    def entries(self) -> list[MemoryEntry]:
        return self._entries

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def embedding_model(self) -> str:
        return self._embedding_model

    @property
    def dim(self) -> int:
        return self._dim

    def add(self, sign_id: str, image_path: str) -> MemoryEntry:
        """Append a new entry, assigning the next positional ``vector_id``."""
        entry = MemoryEntry(
            vector_id=len(self._entries),
            sign_id=sign_id,
            image_path=image_path,
            embedding_model=self._embedding_model,
            created_time=datetime.now(timezone.utc).isoformat(),
        )
        self._entries.append(entry)
        return entry

    def get(self, vector_id: int) -> MemoryEntry:
        return self._entries[vector_id]

    def reset(self) -> None:
        self._entries = []

    def save(self, path: str | Path) -> None:
        """Write the store to ``path`` as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedding_model": self._embedding_model,
            "dim": self._dim,
            "updated_time": datetime.now(timezone.utc).isoformat(),
            "entries": [asdict(entry) for entry in self._entries],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("metadata_saved", path=str(path), entries=self.size)

    def load(self, path: str | Path) -> None:
        """Replace current contents with the store persisted at ``path``."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Metadata not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._embedding_model = payload.get("embedding_model", self._embedding_model)
        self._dim = int(payload.get("dim", self._dim))
        self._entries = [MemoryEntry(**entry) for entry in payload.get("entries", [])]
        logger.info("metadata_loaded", path=str(path), entries=self.size)
