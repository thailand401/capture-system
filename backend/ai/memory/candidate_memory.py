"""Candidate Memory: the mutable, unverified layer of the two-tier memory.

New high-confidence embeddings discovered at inference time land here first.
On append the memory:

1. Rejects near-identical duplicates (cosine > ``duplicate_threshold``).
2. Merges the sighting into the nearest existing candidate cluster when the
   cosine similarity is >= ``match_threshold`` (same physical sign seen
   again — this is what advances the day/device promotion counters).
3. Otherwise starts a brand-new candidate cluster.

Vectors are expected to be L2-normalized so inner product == cosine.
Persistence is a self-contained JSON file (embeddings inline) so the layer
is portable and easy to export/import.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from ai.memory.observation import Candidate, Observation
from ai.utils.logging import get_logger

if TYPE_CHECKING:
    import numpy as np

logger = get_logger("memory.candidate")


class AppendStatus(str, Enum):
    """Outcome of a candidate append."""

    CREATED = "created"
    MERGED = "merged"
    DUPLICATE = "duplicate"


@dataclass(frozen=True, slots=True)
class AppendResult:
    """What happened when an embedding was appended to candidate memory."""

    status: AppendStatus
    candidate_id: str
    similarity: float


class CandidateMemory:
    """Cluster-based store of unverified candidate embeddings."""

    def __init__(
        self,
        *,
        embedding_model: str,
        dim: int,
        duplicate_threshold: float = 0.995,
        match_threshold: float = 0.95,
    ) -> None:
        self._embedding_model = embedding_model
        self._dim = dim
        self._duplicate_threshold = duplicate_threshold
        self._match_threshold = match_threshold
        self._candidates: list[Candidate] = []
        self._total_attempts = 0
        self._duplicates = 0
        self._promoted = 0

    # --- Introspection ---------------------------------------------------

    @property
    def candidates(self) -> list[Candidate]:
        return self._candidates

    @property
    def size(self) -> int:
        """Number of candidate clusters."""
        return len(self._candidates)

    @property
    def observation_count(self) -> int:
        return sum(c.observation_count for c in self._candidates)

    @property
    def total_attempts(self) -> int:
        return self._total_attempts

    @property
    def duplicates_ignored(self) -> int:
        return self._duplicates

    @property
    def promoted_count(self) -> int:
        return self._promoted

    def get(self, candidate_id: str) -> Candidate | None:
        return next((c for c in self._candidates if c.candidate_id == candidate_id), None)

    # --- Mutations -------------------------------------------------------

    def append(self, embedding: "np.ndarray", observation: Observation) -> AppendResult:
        """Append an embedding + observation, applying duplicate/merge rules."""
        import numpy as np

        self._total_attempts += 1
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)

        nearest, similarity = self._nearest(vector)
        if nearest is not None and similarity > self._duplicate_threshold:
            self._duplicates += 1
            logger.info("candidate_duplicate_ignored", candidate_id=nearest.candidate_id, sim=round(similarity, 4))
            return AppendResult(AppendStatus.DUPLICATE, nearest.candidate_id, similarity)

        if nearest is not None and similarity >= self._match_threshold:
            nearest.add(observation, vector)
            logger.info("candidate_merged", candidate_id=nearest.candidate_id, sim=round(similarity, 4))
            return AppendResult(AppendStatus.MERGED, nearest.candidate_id, similarity)

        candidate = Candidate()
        candidate.add(observation, vector)
        self._candidates.append(candidate)
        logger.info("candidate_created", candidate_id=candidate.candidate_id)
        return AppendResult(AppendStatus.CREATED, candidate.candidate_id, similarity)

    def remove(self, candidate_id: str) -> bool:
        """Drop a candidate cluster (e.g. after promotion)."""
        before = len(self._candidates)
        self._candidates = [c for c in self._candidates if c.candidate_id != candidate_id]
        return len(self._candidates) < before

    def verify(self, candidate_id: str) -> bool:
        """Mark a candidate as human-verified (promotion rule #3)."""
        candidate = self.get(candidate_id)
        if candidate is None:
            return False
        candidate.human_verified = True
        return True

    def record_promotion(self, count: int = 1) -> None:
        self._promoted += count

    # --- Search ----------------------------------------------------------

    def search(self, embedding: "np.ndarray", k: int = 5) -> list[tuple[Candidate, float]]:
        """Return up to ``k`` nearest candidates with their cosine similarity."""
        import numpy as np

        if not self._candidates:
            return []
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        scored = [(c, float(np.dot(c.representative(), vector))) for c in self._candidates]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def _nearest(self, vector: "np.ndarray") -> tuple[Candidate | None, float]:
        results = self.search(vector, k=1)
        return results[0] if results else (None, -1.0)

    # --- Persistence -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "embedding_model": self._embedding_model,
                "dim": self._dim,
                "total_attempts": self._total_attempts,
                "duplicates": self._duplicates,
                "promoted": self._promoted,
            },
            "candidates": [
                {
                    "candidate_id": c.candidate_id,
                    "human_verified": c.human_verified,
                    "observations": [o.model_dump() for o in c.observations],
                    "vectors": [v.astype(float).tolist() for v in c.vectors],
                }
                for c in self._candidates
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("candidate_memory_saved", path=str(path), candidates=self.size)

    def load(self, path: str | Path) -> None:
        import numpy as np

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Candidate memory not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta = payload.get("meta", {})
        self._embedding_model = meta.get("embedding_model", self._embedding_model)
        self._dim = int(meta.get("dim", self._dim))
        self._total_attempts = int(meta.get("total_attempts", 0))
        self._duplicates = int(meta.get("duplicates", 0))
        self._promoted = int(meta.get("promoted", 0))

        self._candidates = []
        for raw in payload.get("candidates", []):
            candidate = Candidate(
                candidate_id=raw["candidate_id"],
                observations=[Observation(**o) for o in raw.get("observations", [])],
                vectors=[np.asarray(v, dtype=np.float32) for v in raw.get("vectors", [])],
                human_verified=bool(raw.get("human_verified", False)),
            )
            self._candidates.append(candidate)
        logger.info("candidate_memory_loaded", path=str(path), candidates=self.size)
