"""Voting engine unit tests."""

from __future__ import annotations

from ai.matcher.voting_engine import VotingEngine
from ai.memory.memory_manager import MemoryMatch


def _match(sign_id: str, similarity: float) -> MemoryMatch:
    return MemoryMatch(sign_id=sign_id, image_path=f"{sign_id}.png", similarity=similarity, vector_id=0)


def test_majority_sign_wins():
    matches = [
        _match("102", 0.91),
        _match("102", 0.90),
        _match("102", 0.88),
        _match("103", 0.87),
        _match("102", 0.86),
        _match("104", 0.80),
        _match("102", 0.79),
    ]
    result = VotingEngine().vote(matches)
    assert result.sign_id == "102"
    assert 0.0 < result.confidence <= 1.0
    assert result.votes["102"] == 5


def test_empty_matches_returns_none():
    result = VotingEngine().vote([])
    assert result.sign_id is None
    assert result.confidence == 0.0


def test_high_similarity_minority_can_win_on_weight():
    # One extremely strong match should be able to beat a weak majority.
    matches = [
        _match("200", 0.99),
        _match("201", 0.10),
        _match("201", 0.11),
    ]
    result = VotingEngine(majority_weight=0.1, average_weight=0.45, weighted_weight=0.45).vote(matches)
    assert result.sign_id == "200"
