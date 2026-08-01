"""Voting engine: decide the winning sign id from top-K memory hits.

The top-K neighbours often mix a few sign ids, e.g.::

    102 102 102 103 102 104 102

The winner is chosen by combining three signals per candidate sign:

* **Majority**      — how many of the K hits belong to the sign.
* **Average**       — mean similarity of that sign's hits.
* **Weighted**      — the sign's share of the total similarity mass.

These are blended into a single score; the highest-scoring sign wins and a
normalized ``confidence`` in ``[0, 1]`` is returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ai.memory.memory_manager import MemoryMatch
from ai.utils.logging import get_logger

logger = get_logger("matcher.voting")


@dataclass(frozen=True, slots=True)
class VoteResult:
    """Outcome of voting over the top-K matches."""

    sign_id: str | None
    confidence: float
    representative_similarity: float
    votes: dict[str, int] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)


class VotingEngine:
    """Blends majority, average and weighted-similarity voting."""

    def __init__(
        self,
        *,
        majority_weight: float = 0.34,
        average_weight: float = 0.33,
        weighted_weight: float = 0.33,
    ) -> None:
        total = majority_weight + average_weight + weighted_weight
        if total <= 0:
            raise ValueError("Voting weights must sum to a positive value.")
        self._w_majority = majority_weight / total
        self._w_average = average_weight / total
        self._w_weighted = weighted_weight / total

    def vote(self, matches: list[MemoryMatch]) -> VoteResult:
        """Return the winning sign and a confidence for ``matches``."""
        if not matches:
            return VoteResult(sign_id=None, confidence=0.0, representative_similarity=0.0)

        k = len(matches)
        # Similarities are cosine in [-1, 1]; clamp negatives so they don't
        # subtract from a sign's similarity mass.
        clamped = {id(m): max(0.0, m.similarity) for m in matches}
        total_similarity = sum(clamped.values()) or 1.0

        votes: dict[str, int] = {}
        sum_similarity: dict[str, float] = {}
        max_similarity: dict[str, float] = {}
        for match in matches:
            sign = match.sign_id
            votes[sign] = votes.get(sign, 0) + 1
            sum_similarity[sign] = sum_similarity.get(sign, 0.0) + clamped[id(match)]
            max_similarity[sign] = max(max_similarity.get(sign, 0.0), match.similarity)

        scores: dict[str, float] = {}
        for sign in votes:
            majority = votes[sign] / k
            average = sum_similarity[sign] / votes[sign]
            weighted = sum_similarity[sign] / total_similarity
            scores[sign] = (
                self._w_majority * majority
                + self._w_average * average
                + self._w_weighted * weighted
            )

        best_sign = max(scores, key=lambda s: scores[s])
        confidence = max(0.0, min(1.0, scores[best_sign]))

        logger.info(
            "vote_completed",
            winner=best_sign,
            confidence=round(confidence, 4),
            votes=votes.get(best_sign),
            k=k,
        )
        return VoteResult(
            sign_id=best_sign,
            confidence=confidence,
            representative_similarity=max_similarity[best_sign],
            votes=votes,
            scores=scores,
        )
