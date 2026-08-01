"""MemoryManager unit tests (build / append / search / persistence)."""

from __future__ import annotations

import numpy as np
import pytest

from ai.memory import memory_manager as memory_manager_module
from ai.memory.memory_manager import MemoryManager
from ai.memory.vector_store import VectorStore
from ai.models.embedding import Embedding
from tests_ai.fakes import FakeIndex

DIM = 8


def _color_for(sign_id: str) -> np.ndarray:
    """Deterministic solid-color image per sign id (same sign -> same pixels)."""
    value = (abs(hash(sign_id)) % 200) + 30
    return np.full((16, 16, 3), value, dtype=np.uint8)


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    """Create dataset/<sign>/<n>.png files and stub image loading."""
    root = tmp_path / "dataset"
    layout = {"102": 3, "103a": 2, "127": 2}
    for sign_id, count in layout.items():
        sign_dir = root / sign_id
        sign_dir.mkdir(parents=True)
        for i in range(count):
            (sign_dir / f"{i}.png").write_bytes(b"stub")

    def fake_load(path):
        return _color_for(str(path).split("/")[-2])

    monkeypatch.setattr(memory_manager_module, "load_image", fake_load)
    return root, layout


def test_build_indexes_every_image(memory, dataset):
    root, layout = dataset
    total = memory.build()
    assert total == sum(layout.values())
    assert memory.exists()


def test_search_returns_matching_sign(memory, dataset):
    memory.build()
    query = memory.embedding_engine.encode(_color_for("127"))
    matches = memory.search(query, k=5)
    assert matches
    assert matches[0].sign_id == "127"


def test_append_grows_memory_without_rebuild(memory, dataset, monkeypatch):
    memory.build()
    original = len(memory._store.entries)  # noqa: SLF001 - test introspection
    match = memory.append(image_path="dataset/999/0.png", sign_id="999", persist=False)
    assert match.sign_id == "999"
    assert len(memory._store.entries) == original + 1  # noqa: SLF001


def test_persistence_round_trip(memory, dataset, embedding_engine, tmp_path):
    memory.build()
    reloaded = MemoryManager(
        embedding_engine=embedding_engine,
        index=FakeIndex(dim=DIM),
        store=VectorStore(embedding_model="fake-encoder", dim=DIM),
        dataset_dir=tmp_path / "dataset",
        index_path=tmp_path / "memory" / "vectors.faiss",
        metadata_path=tmp_path / "memory" / "metadata.json",
    )
    reloaded.load()
    query = memory.embedding_engine.encode(_color_for("102"))
    matches = reloaded.search(query, k=3)
    assert matches[0].sign_id == "102"


def test_search_empty_memory_returns_empty(memory):
    empty = Embedding(vector=np.zeros(DIM, dtype=np.float32), model_name="fake-encoder")
    assert memory.search(empty, k=5) == []
