"""Candidate memory unit tests: duplicate detection + clustering."""

from __future__ import annotations

import numpy as np

from ai.memory.candidate_memory import AppendStatus, CandidateMemory
from ai.memory.observation import Observation

DIM = 4


def _norm(values) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _obs(sign: str = "102", day: str = "2026-01-01", device: str = "d1", score: float = 95.0) -> Observation:
    return Observation(
        traffic_sign_id=sign,
        timestamp=f"{day}T10:00:00+00:00",
        device_id=device,
        confidence=0.9,
        validation_score=score,
        blur_score=0.9,
        image_hash="hash",
        image_path=f"{sign}.png",
    )


def _memory() -> CandidateMemory:
    return CandidateMemory(embedding_model="fake", dim=DIM)


def test_first_append_creates_candidate():
    memory = _memory()
    result = memory.append(_norm([1, 0, 0, 0]), _obs())
    assert result.status is AppendStatus.CREATED
    assert memory.size == 1


def test_near_identical_is_ignored_as_duplicate():
    memory = _memory()
    memory.append(_norm([1, 0, 0, 0]), _obs(day="2026-01-01"))
    result = memory.append(_norm([1, 0, 0, 0]), _obs(day="2026-01-02"))
    assert result.status is AppendStatus.DUPLICATE
    assert memory.size == 1
    assert memory.duplicates_ignored == 1
    assert memory.observation_count == 1  # duplicate observation not recorded


def test_similar_sighting_merges_and_counts_days_devices():
    memory = _memory()
    memory.append(_norm([1, 0, 0, 0]), _obs(day="2026-01-01", device="d1"))
    result = memory.append(_norm([0.96, 0.28, 0, 0]), _obs(day="2026-01-02", device="d2"))
    assert result.status is AppendStatus.MERGED
    assert memory.size == 1
    candidate = memory.candidates[0]
    assert candidate.distinct_days == 2
    assert candidate.distinct_devices == 2


def test_dissimilar_creates_new_candidate():
    memory = _memory()
    memory.append(_norm([1, 0, 0, 0]), _obs())
    result = memory.append(_norm([0, 1, 0, 0]), _obs())
    assert result.status is AppendStatus.CREATED
    assert memory.size == 2


def test_persistence_round_trip(tmp_path):
    memory = _memory()
    memory.append(_norm([1, 0, 0, 0]), _obs(day="2026-01-01"))
    memory.append(_norm([0, 1, 0, 0]), _obs(day="2026-01-02", device="d2"))
    path = tmp_path / "candidates.json"
    memory.save(path)

    reloaded = _memory()
    reloaded.load(path)
    assert reloaded.size == 2
    assert reloaded.observation_count == 2
    assert reloaded.total_attempts == memory.total_attempts


def test_verify_sets_flag():
    memory = _memory()
    result = memory.append(_norm([1, 0, 0, 0]), _obs())
    assert memory.verify(result.candidate_id) is True
    assert memory.get(result.candidate_id).human_verified is True
