"""SimilaritySearch unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from ai.matcher.similarity_search import SimilaritySearch
from ai.memory import memory_manager as memory_manager_module


@pytest.fixture
def populated_memory(memory, monkeypatch):
    def fake_load(path):
        value = (abs(hash(str(path).split("/")[-2])) % 200) + 30
        return np.full((16, 16, 3), value, dtype=np.uint8)

    monkeypatch.setattr(memory_manager_module, "load_image", fake_load)
    for sign_id in ("102", "103", "104"):
        memory.append(image_path=f"dataset/{sign_id}/0.png", sign_id=sign_id, persist=False)
    return memory


def test_uses_default_k(populated_memory):
    search = SimilaritySearch(memory=populated_memory, default_k=2)
    query = populated_memory.embedding_engine.encode(
        np.full((16, 16, 3), (abs(hash("102")) % 200) + 30, dtype=np.uint8)
    )
    matches = search.search(query)
    assert len(matches) == 2
    assert matches[0].sign_id == "102"


def test_explicit_k_overrides_default(populated_memory):
    search = SimilaritySearch(memory=populated_memory, default_k=1)
    query = populated_memory.embedding_engine.encode(np.full((16, 16, 3), 40, dtype=np.uint8))
    assert len(search.search(query, k=3)) == 3
