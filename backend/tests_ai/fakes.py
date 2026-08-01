"""Test doubles for the recognition engine.

These fakes let the engine's orchestration and memory logic be tested with
only numpy installed — no torch, faiss, ultralytics or OpenCV required.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ai.detector.detector import Detection
from ai.embedding.embedding_engine import ImageEncoder
from ai.memory.faiss_index import VectorIndex
from ai.models.bbox import BoundingBox
from ai.validator.blur_validator import BlurResult
from ai.validator.color_validator import ColorResult
from ai.validator.shape_validator import ShapeResult


class FakeEncoder(ImageEncoder):
    """Deterministic encoder: identical images -> identical vectors."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    @property
    def model_name(self) -> str:
        return "fake-encoder"

    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        rows = []
        for img in images:
            arr = np.ascontiguousarray(img)
            if arr.size == 0:
                rows.append(np.zeros(self._dim))
                continue
            # Hash the pixels to a seed: identical images -> identical vectors,
            # distinct images -> well-separated random directions.
            digest = hashlib.blake2b(arr.tobytes(), digest_size=8).digest()
            seed = int.from_bytes(digest, "little")
            rows.append(np.random.default_rng(seed).standard_normal(self._dim))
        return np.asarray(rows, dtype=np.float32)


class FakeIndex(VectorIndex):
    """Pure-numpy brute-force inner-product index (FAISS stand-in)."""

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._vectors = np.zeros((0, dim), dtype=np.float32)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def size(self) -> int:
        return int(self._vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        batch = np.asarray(vectors, dtype=np.float32).reshape(-1, self._dim)
        self._vectors = np.vstack([self._vectors, batch])

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(queries, dtype=np.float32).reshape(-1, self._dim)
        if self.size == 0:
            empty = np.full((q.shape[0], k), -1)
            return np.zeros((q.shape[0], k), dtype=np.float32), empty
        sims = q @ self._vectors.T
        k = min(k, self.size)
        ids = np.argsort(-sims, axis=1)[:, :k]
        scores = np.take_along_axis(sims, ids, axis=1)
        return scores.astype(np.float32), ids

    def reconstruct_all(self) -> np.ndarray:
        return self._vectors.copy()

    def reset(self) -> None:
        self._vectors = np.zeros((0, self._dim), dtype=np.float32)

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Write to the exact path (np.save on a handle does not append .npy).
        with open(path, "wb") as handle:
            np.save(handle, self._vectors)

    def load(self, path) -> None:
        with open(path, "rb") as handle:
            self._vectors = np.load(handle, allow_pickle=False)


class FakeDetector:
    """Returns one full-frame detection so the pipeline always has a crop."""

    def __init__(self, confidence: float = 0.9) -> None:
        self._confidence = confidence

    def detect(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        return [Detection(bbox=BoundingBox(0, 0, float(w), float(h)), confidence=self._confidence)]


class FakeShapeValidator:
    def validate(self, crop: np.ndarray) -> ShapeResult:
        return ShapeResult(shape="circle", score=0.8, vertices=0)


class FakeColorValidator:
    def validate(self, crop: np.ndarray) -> ColorResult:
        return ColorResult(dominant_colors=["red"], score=0.7, fractions={"red": 0.7})


class FakeBlurValidator:
    def validate(self, crop: np.ndarray) -> BlurResult:
        return BlurResult(score=0.9, variance=300.0, is_blurry=False)
