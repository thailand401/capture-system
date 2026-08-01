"""Embedding engine unit tests (uses the deterministic fake encoder)."""

from __future__ import annotations

import numpy as np

from ai.embedding.embedding_engine import EmbeddingEngine
from ai.embedding.feature_normalizer import FeatureNormalizer
from tests_ai.fakes import FakeEncoder

DIM = 8


def test_embedding_is_l2_normalized(embedding_engine: EmbeddingEngine, sign_image):
    embedding = embedding_engine.encode(sign_image(seed=1))
    norm = float(np.linalg.norm(embedding.vector))
    assert embedding.dim == DIM
    assert abs(norm - 1.0) < 1e-5


def test_identical_images_give_identical_embeddings(embedding_engine, sign_image):
    a = embedding_engine.encode(sign_image(seed=7))
    b = embedding_engine.encode(sign_image(seed=7))
    assert np.allclose(a.vector, b.vector)


def test_different_images_differ(embedding_engine, sign_image):
    a = embedding_engine.encode(sign_image(seed=1))
    b = embedding_engine.encode(sign_image(seed=2))
    assert not np.allclose(a.vector, b.vector)


def test_normalizer_handles_zero_vector():
    normalized = FeatureNormalizer().normalize(np.zeros((1, DIM), dtype=np.float32))
    assert np.all(np.isfinite(normalized))
    assert np.allclose(normalized, 0.0)


def test_model_name_propagates():
    engine = EmbeddingEngine(encoder=FakeEncoder(dim=DIM))
    assert engine.model_name == "fake-encoder"
