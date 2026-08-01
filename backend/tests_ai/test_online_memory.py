"""OnlineMemoryManager tests: promote / prune / rollback / export / stats."""

from __future__ import annotations

import numpy as np

from ai.config import OnlineLearningConfig
from ai.embedding.embedding_engine import EmbeddingEngine
from ai.memory.candidate_memory import CandidateMemory
from ai.memory.memory_manager import MemoryManager
from ai.memory.memory_optimizer import MemoryOptimizer
from ai.memory.observation import Candidate, Observation
from ai.memory.online_memory import OnlineMemoryManager
from ai.memory.promotion_engine import PromotionEngine
from ai.memory.vector_store import VectorStore
from ai.models.embedding import Embedding
from ai.models.prediction import Prediction
from tests_ai.fakes import FakeEncoder, FakeIndex

DIM = 4


def _norm(values) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _observation(day: str, device: str) -> Observation:
    return Observation(
        traffic_sign_id="102",
        timestamp=f"{day}T00:00:00+00:00",
        device_id=device,
        confidence=0.9,
        validation_score=95.0,
        blur_score=0.9,
        image_hash="hash",
        image_path="102.png",
    )


def _eligible_candidate() -> Candidate:
    candidate = Candidate()
    for i in range(5):  # 5 distinct days -> promotion rule #1
        candidate.add(_observation(day=f"2026-01-0{i + 1}", device="d1"), _norm([1, 0, 0, 0]))
    return candidate


def _make_online(tmp_path, *, config: OnlineLearningConfig | None = None) -> OnlineMemoryManager:
    memory_dir = tmp_path / "memory"
    permanent = MemoryManager(
        embedding_engine=EmbeddingEngine(encoder=FakeEncoder(dim=DIM)),
        index=FakeIndex(dim=DIM),
        store=VectorStore(embedding_model="fake", dim=DIM),
        dataset_dir=tmp_path / "dataset",
        index_path=memory_dir / "vectors.faiss",
        metadata_path=memory_dir / "metadata.json",
    )
    return OnlineMemoryManager(
        permanent=permanent,
        candidates=CandidateMemory(embedding_model="fake", dim=DIM),
        promotion_engine=PromotionEngine(min_distinct_days=5, min_distinct_devices=3),
        optimizer=MemoryOptimizer(redundancy_threshold=0.98),
        config=config or OnlineLearningConfig(),
        memory_dir=memory_dir,
        candidates_path=memory_dir / "candidates.json",
        version_path=memory_dir / "version.json",
    )


def test_observe_gate_appends_only_high_quality(tmp_path):
    online = _make_online(tmp_path)
    embedding = Embedding(vector=_norm([1, 0, 0, 0]), model_name="fake")

    good = Prediction(traffic_sign_id="102", similarity=0.97, validation_score=95.0, bbox=(0, 0, 1, 1))
    online.observe(prediction=good, embedding=embedding, image_hash="h")
    assert online.candidate_memory.size == 1

    low_score = Prediction(traffic_sign_id="102", similarity=0.97, validation_score=80.0, bbox=(0, 0, 1, 1))
    low_sim = Prediction(traffic_sign_id="102", similarity=0.80, validation_score=95.0, bbox=(0, 0, 1, 1))
    online.observe(prediction=low_score, embedding=embedding, image_hash="h")
    online.observe(prediction=low_sim, embedding=embedding, image_hash="h")
    assert online.candidate_memory.size == 1  # neither passed the gate


def test_promote_moves_candidate_to_permanent(tmp_path):
    online = _make_online(tmp_path)
    online.candidate_memory.candidates.append(_eligible_candidate())

    result = online.promote()
    assert len(result.promoted_ids) == 1
    assert online.permanent.store.size == 1
    assert online.candidate_memory.size == 0
    assert online.statistics().num_promoted == 1


def test_rollback_restores_pre_promote_state(tmp_path):
    online = _make_online(tmp_path)
    online.candidate_memory.candidates.append(_eligible_candidate())

    online.promote()
    assert online.permanent.store.size == 1

    online.rollback()
    assert online.permanent.store.size == 0
    assert online.candidate_memory.size == 1


def test_prune_removes_redundant_permanent_vectors(tmp_path):
    online = _make_online(tmp_path)
    online.permanent.append_vector(_norm([1, 0, 0, 0]), sign_id="102", image_path="a.png", persist=False)
    online.permanent.append_vector(_norm([1, 0, 0, 0]), sign_id="102", image_path="b.png", persist=False)
    online.permanent.append_vector(_norm([0, 1, 0, 0]), sign_id="102", image_path="c.png", persist=False)
    online.save()

    removed = online.prune()
    assert removed == 1
    assert online.permanent.store.size == 2


def test_export_import_round_trip(tmp_path):
    online = _make_online(tmp_path)
    online.permanent.append_vector(_norm([1, 0, 0, 0]), sign_id="102", image_path="a.png", persist=False)
    online.candidate_memory.candidates.append(_eligible_candidate())
    online.save()

    bundle = online.export(tmp_path / "bundle.zip")

    other = _make_online(tmp_path / "other")
    other.import_(bundle)
    assert other.permanent.store.size == 1
    assert other.candidate_memory.size == 1


def test_statistics_reports_duplicate_ratio(tmp_path):
    online = _make_online(tmp_path)
    embedding = Embedding(vector=_norm([1, 0, 0, 0]), model_name="fake")
    good = Prediction(traffic_sign_id="102", similarity=0.97, validation_score=95.0, bbox=(0, 0, 1, 1))
    online.observe(prediction=good, embedding=embedding, image_hash="h")
    online.observe(prediction=good, embedding=embedding, image_hash="h")  # duplicate

    stats = online.statistics()
    assert stats.num_candidates == 1
    assert 0.0 < stats.duplicate_ratio <= 1.0
    assert 0.0 <= stats.recognition_accuracy_estimate <= 1.0
