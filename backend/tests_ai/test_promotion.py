"""Promotion engine + memory optimizer unit tests."""

from __future__ import annotations

import numpy as np

from ai.memory.memory_optimizer import MemoryOptimizer
from ai.memory.observation import Candidate, Observation
from ai.memory.promotion_engine import PromotionEngine


def _make_candidate(*, days: int, devices: int, verified: bool = False) -> Candidate:
    candidate = Candidate(human_verified=verified)
    count = max(days, devices, 1)
    for i in range(count):
        day = (i % days) + 1 if days else 1
        device = (i % devices) + 1 if devices else 1
        observation = Observation(
            traffic_sign_id="102",
            timestamp=f"2026-01-{day:02d}T00:00:00+00:00",
            device_id=f"device-{device}",
            confidence=0.9,
            validation_score=95.0,
            blur_score=0.9,
            image_hash="hash",
            image_path="102.png",
        )
        candidate.add(observation, np.asarray([1, 0, 0, 0], dtype=np.float32))
    return candidate


def test_promote_on_distinct_days():
    engine = PromotionEngine(min_distinct_days=5, min_distinct_devices=3)
    assert engine.is_eligible(_make_candidate(days=5, devices=1)) is True
    assert engine.reason(_make_candidate(days=5, devices=1)) == "distinct_days"


def test_promote_on_distinct_devices():
    engine = PromotionEngine(min_distinct_days=5, min_distinct_devices=3)
    assert engine.is_eligible(_make_candidate(days=1, devices=3)) is True


def test_promote_on_human_verified():
    engine = PromotionEngine()
    assert engine.is_eligible(_make_candidate(days=1, devices=1, verified=True)) is True
    assert engine.reason(_make_candidate(days=1, devices=1, verified=True)) == "human_verified"


def test_not_eligible_below_thresholds():
    engine = PromotionEngine(min_distinct_days=5, min_distinct_devices=3)
    assert engine.is_eligible(_make_candidate(days=2, devices=2)) is False


def test_select_returns_only_eligible():
    engine = PromotionEngine(min_distinct_days=5, min_distinct_devices=3)
    candidates = [_make_candidate(days=5, devices=1), _make_candidate(days=1, devices=1)]
    assert engine.select(candidates) == [candidates[0]]


def _norm(values) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def test_optimizer_removes_same_sign_duplicates():
    optimizer = MemoryOptimizer(redundancy_threshold=0.98)
    vectors = np.stack([_norm([1, 0]), _norm([1, 0]), _norm([0, 1])])
    keep = optimizer.select_representatives(vectors, ["102", "102", "102"])
    assert keep == [0, 2]


def test_optimizer_keeps_different_signs():
    optimizer = MemoryOptimizer(redundancy_threshold=0.98)
    vectors = np.stack([_norm([1, 0]), _norm([1, 0])])
    keep = optimizer.select_representatives(vectors, ["102", "103"])
    assert keep == [0, 1]
