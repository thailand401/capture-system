"""Shared fixtures for the AI-engine test suite.

Guards on numpy so the whole suite is skipped (rather than erroring) in an
environment where the CV/ML stack has not been installed yet.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("loguru")

import numpy as np  # noqa: E402

from ai.embedding.embedding_engine import EmbeddingEngine  # noqa: E402
from ai.memory.memory_manager import MemoryManager  # noqa: E402
from ai.memory.vector_store import VectorStore  # noqa: E402
from tests_ai.fakes import FakeEncoder, FakeIndex  # noqa: E402

DIM = 8


@pytest.fixture
def embedding_engine() -> EmbeddingEngine:
    return EmbeddingEngine(encoder=FakeEncoder(dim=DIM))


@pytest.fixture
def sign_image():
    """Factory building a deterministic solid-color image for a sign id."""

    def _make(seed: int, size: int = 16) -> np.ndarray:
        rng = np.random.default_rng(seed)
        color = rng.integers(0, 255, size=3)
        return np.full((size, size, 3), color, dtype=np.uint8)

    return _make


@pytest.fixture
def memory(embedding_engine, tmp_path):
    return MemoryManager(
        embedding_engine=embedding_engine,
        index=FakeIndex(dim=DIM),
        store=VectorStore(embedding_model="fake-encoder", dim=DIM),
        dataset_dir=tmp_path / "dataset",
        index_path=tmp_path / "memory" / "vectors.faiss",
        metadata_path=tmp_path / "memory" / "metadata.json",
    )
