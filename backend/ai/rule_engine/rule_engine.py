"""Rule engine: fuse every stage signal into a single 0..100 score.

Combines YOLO confidence, embedding similarity, voting confidence and the
shape / color / blur scores using configurable weights. All inputs are
normalized to ``[0, 1]`` first; the weighted sum is scaled to ``0..100``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ai.config import RuleWeights
from ai.utils.logging import get_logger

logger = get_logger("rule_engine")


@dataclass(frozen=True, slots=True)
class RuleInputs:
    """The per-stage signals fused into the final validation score."""

    yolo_confidence: float
    embedding_similarity: float
    voting_confidence: float
    shape_score: float
    color_score: float
    blur_score: float


class RuleEngine:
    """Weighted fusion of stage signals into a ``0..100`` validation score."""

    def __init__(self, *, weights: RuleWeights | None = None) -> None:
        self._weights = (weights or RuleWeights()).normalized()

    def evaluate(self, inputs: RuleInputs) -> float:
        """Return a ``0..100`` validation score for ``inputs``."""
        # Cosine similarity is in [-1, 1]; map to [0, 1] before weighting.
        similarity = (inputs.embedding_similarity + 1.0) / 2.0

        signals = {
            "yolo": inputs.yolo_confidence,
            "similarity": similarity,
            "voting": inputs.voting_confidence,
            "shape": inputs.shape_score,
            "color": inputs.color_score,
            "blur": inputs.blur_score,
        }

        score01 = sum(self._weights[name] * _clamp01(value) for name, value in signals.items())
        score = round(100.0 * _clamp01(score01), 2)
        logger.info("validation_scored", score=score, signals={k: round(v, 3) for k, v in signals.items()})
        return score


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
