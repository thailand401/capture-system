"""Online Memory Manager: coordinates the two-layer learning memory.

Ties together Permanent Memory (verified, the recognition source of truth),
Candidate Memory (unverified discoveries), the Promotion Engine and the
Memory Optimizer, and adds versioning / rollback / export / import.

Public API (as specified): ``append``, ``search``, ``promote``,
``rollback``, ``prune``, ``export``, ``import_`` and ``statistics`` (plus
``verify`` for human confirmation and lifecycle helpers).
"""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ai.config import OnlineLearningConfig
from ai.memory.candidate_memory import AppendResult, CandidateMemory
from ai.memory.memory_manager import MemoryManager
from ai.memory.memory_optimizer import MemoryOptimizer
from ai.memory.observation import Observation, ObservationContext
from ai.memory.promotion_engine import PromotionEngine
from ai.models.embedding import Embedding
from ai.models.prediction import Prediction
from ai.utils.logging import get_logger

if TYPE_CHECKING:
    import numpy as np

logger = get_logger("memory.online")


class MemoryStatistics(BaseModel):
    """A snapshot of memory health / learning progress."""

    num_vectors: int
    num_candidates: int
    num_candidate_observations: int
    num_promoted: int
    memory_size_bytes: int
    duplicate_ratio: float
    recognition_accuracy_estimate: float
    memory_version: int


@dataclass(frozen=True, slots=True)
class PromotionResult:
    """Which candidates were promoted into permanent memory."""

    promoted_ids: list[str]
    reasons: dict[str, str]


class OnlineMemoryManager:
    """Two-layer memory with online learning, promotion, pruning and versioning."""

    def __init__(
        self,
        *,
        permanent: MemoryManager,
        candidates: CandidateMemory,
        promotion_engine: PromotionEngine,
        optimizer: MemoryOptimizer,
        config: OnlineLearningConfig,
        memory_dir: str | Path = "memory",
        candidates_path: str | Path = "memory/candidates.json",
        version_path: str | Path = "memory/version.json",
    ) -> None:
        self._permanent = permanent
        self._candidates = candidates
        self._promotion = promotion_engine
        self._optimizer = optimizer
        self._config = config
        self._memory_dir = Path(memory_dir)
        self._candidates_path = Path(candidates_path)
        self._version_path = Path(version_path)
        self._versions_dir = self._memory_dir / "versions"
        self._version_state = {"version": 0, "history": []}

    # --- Introspection ---------------------------------------------------

    @property
    def permanent(self) -> MemoryManager:
        return self._permanent

    @property
    def candidate_memory(self) -> CandidateMemory:
        return self._candidates

    @property
    def version(self) -> int:
        return int(self._version_state["version"])

    # --- Lifecycle -------------------------------------------------------

    def ensure_ready(self) -> None:
        """Load permanent + candidate memory, building permanent if missing."""
        self._permanent.ensure_ready()
        if self._candidates_path.exists():
            self._candidates.load(self._candidates_path)
        self._load_version_state()

    def load_learning_state(self) -> None:
        """Load candidate memory + version history only (permanent left as-is).

        Used when the permanent layer was already built/loaded elsewhere and
        only the online-learning state needs restoring.
        """
        if self._candidates_path.exists():
            self._candidates.load(self._candidates_path)
        self._load_version_state()

    def save(self) -> None:
        """Persist both layers (permanent index/metadata + candidates)."""
        self._permanent.save()
        self._candidates.save(self._candidates_path)

    # --- Learning loop ---------------------------------------------------

    def observe(
        self,
        *,
        prediction: Prediction,
        embedding: Embedding,
        image_hash: str,
        context: ObservationContext | None = None,
    ) -> AppendResult | None:
        """Apply the append gate and record a candidate observation.

        Gate: ``validation_score >= min_validation_score`` **and**
        ``similarity >= min_similarity``. Nothing is ever written directly to
        permanent memory here — only to the candidate layer.
        """
        if not self._config.enabled or prediction.traffic_sign_id is None:
            return None
        if prediction.validation_score < self._config.min_validation_score:
            return None
        if prediction.similarity < self._config.min_similarity:
            return None

        observation = Observation.from_context(
            traffic_sign_id=prediction.traffic_sign_id,
            confidence=prediction.yolo_confidence,
            validation_score=prediction.validation_score,
            blur_score=prediction.blur_score,
            image_hash=image_hash,
            image_path=prediction.crop_path or "",
            context=context or ObservationContext(),
        )
        return self.append(embedding.vector, observation)

    def append(self, embedding: "np.ndarray", observation: Observation, *, persist: bool = True) -> AppendResult:
        """Low-level append into candidate memory (duplicate/merge handled there)."""
        result = self._candidates.append(embedding, observation)
        if persist:
            self._candidates.save(self._candidates_path)
        return result

    def search(self, embedding: Embedding, k: int = 20):
        """Search **permanent** memory — the recognition source of truth."""
        return self._permanent.search(embedding, k)

    def verify(self, candidate_id: str, *, persist: bool = True) -> bool:
        """Mark a candidate human-verified (satisfies promotion rule #3)."""
        ok = self._candidates.verify(candidate_id)
        if ok and persist:
            self._candidates.save(self._candidates_path)
        return ok

    # --- Promotion -------------------------------------------------------

    def promote(self) -> PromotionResult:
        """Promote eligible candidates into permanent memory."""
        self._snapshot("pre-promote")
        eligible = self._promotion.select(self._candidates.candidates)

        promoted_ids: list[str] = []
        reasons: dict[str, str] = {}
        for candidate in eligible:
            self._permanent.append_vector(
                candidate.representative(),
                sign_id=candidate.traffic_sign_id,
                image_path=candidate.best_image_path(),
                persist=False,
            )
            reasons[candidate.candidate_id] = self._promotion.reason(candidate) or "unknown"
            self._candidates.remove(candidate.candidate_id)
            promoted_ids.append(candidate.candidate_id)

        if promoted_ids:
            self._candidates.record_promotion(len(promoted_ids))
            self.save()
        logger.info("promotion_done", promoted=len(promoted_ids))
        return PromotionResult(promoted_ids=promoted_ids, reasons=reasons)

    # --- Pruning ---------------------------------------------------------

    def prune(self) -> int:
        """Remove redundant vectors from permanent memory. Returns count removed."""
        self._snapshot("pre-prune")
        vectors = self._permanent.reconstruct_all()
        entries = self._permanent.store.entries
        if len(entries) == 0:
            return 0

        sign_ids = [e.sign_id for e in entries]
        keep = self._optimizer.select_representatives(vectors, sign_ids)
        removed = len(entries) - len(keep)
        if removed == 0:
            return 0

        kept_vectors = vectors[keep]
        kept_meta = [(entries[i].sign_id, entries[i].image_path) for i in keep]
        self._permanent.replace_all(kept_vectors, kept_meta, persist=False)
        self.save()
        logger.info("prune_done", removed=removed, kept=len(keep))
        return removed

    # --- Versioning ------------------------------------------------------

    def rollback(self, version: int | None = None) -> int:
        """Restore memory from a snapshot (default: the most recent)."""
        history = self._version_state["history"]
        if not history:
            raise ValueError("No memory versions available to roll back to.")
        target = history[-1] if version is None else next(
            (h for h in history if h["version"] == version), None
        )
        if target is None:
            raise ValueError(f"Unknown memory version: {version}")

        snap_dir = Path(target["path"])
        for name in ("vectors.faiss", "metadata.json", "candidates.json"):
            source = snap_dir / name
            if source.exists():
                shutil.copy2(source, self._memory_dir / name)
        self._reload_layers()
        logger.info("rollback_done", version=target["version"])
        return int(target["version"])

    def _snapshot(self, label: str) -> int:
        """Persist current state into a new versioned snapshot directory."""
        self.save()
        version = int(self._version_state["version"]) + 1
        snap_dir = self._versions_dir / f"v{version}"
        snap_dir.mkdir(parents=True, exist_ok=True)
        for name in ("vectors.faiss", "metadata.json", "candidates.json"):
            source = self._memory_dir / name
            if source.exists():
                shutil.copy2(source, snap_dir / name)
        self._version_state["version"] = version
        self._version_state["history"].append(
            {
                "version": version,
                "label": label,
                "created": datetime.now(timezone.utc).isoformat(),
                "path": str(snap_dir),
            }
        )
        self._save_version_state()
        logger.info("snapshot_created", version=version, label=label)
        return version

    def _load_version_state(self) -> None:
        if self._version_path.exists():
            self._version_state = json.loads(self._version_path.read_text(encoding="utf-8"))

    def _save_version_state(self) -> None:
        self._version_path.parent.mkdir(parents=True, exist_ok=True)
        self._version_path.write_text(json.dumps(self._version_state, indent=2), encoding="utf-8")

    def _reload_layers(self) -> None:
        self._permanent.load()
        if self._candidates_path.exists():
            self._candidates.load(self._candidates_path)

    # --- Export / import -------------------------------------------------

    def export(self, dest_path: str | Path) -> Path:
        """Bundle permanent + candidate memory into a portable ``.zip``."""
        self.save()
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in ("vectors.faiss", "metadata.json", "candidates.json"):
                source = self._memory_dir / name
                if source.exists():
                    archive.write(source, arcname=name)
        logger.info("memory_exported", path=str(dest))
        return dest

    def import_(self, src_path: str | Path) -> None:
        """Replace current memory with a previously exported bundle."""
        src = Path(src_path)
        if not src.exists():
            raise FileNotFoundError(f"Export bundle not found: {src}")
        self._snapshot("pre-import")
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src, "r") as archive:
            for name in ("vectors.faiss", "metadata.json", "candidates.json"):
                if name in archive.namelist():
                    archive.extract(name, self._memory_dir)
        self._reload_layers()
        logger.info("memory_imported", path=str(src))

    # --- Statistics ------------------------------------------------------

    def statistics(self) -> MemoryStatistics:
        """Compute a health/progress snapshot of the whole memory."""
        attempts = self._candidates.total_attempts
        duplicate_ratio = (self._candidates.duplicates_ignored / attempts) if attempts else 0.0

        scores = [o.validation_score for c in self._candidates.candidates for o in c.observations]
        accuracy_estimate = (sum(scores) / len(scores) / 100.0) if scores else 0.0

        size_bytes = 0
        for name in ("vectors.faiss", "metadata.json", "candidates.json"):
            path = self._memory_dir / name
            if path.exists():
                size_bytes += path.stat().st_size

        return MemoryStatistics(
            num_vectors=self._permanent.store.size,
            num_candidates=self._candidates.size,
            num_candidate_observations=self._candidates.observation_count,
            num_promoted=self._candidates.promoted_count,
            memory_size_bytes=size_bytes,
            duplicate_ratio=round(duplicate_ratio, 4),
            recognition_accuracy_estimate=round(accuracy_estimate, 4),
            memory_version=self.version,
        )
