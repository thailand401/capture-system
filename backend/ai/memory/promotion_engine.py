"""Promotion Engine: decide which candidates earn a place in Permanent Memory.

A candidate is promoted when **any** rule holds:

1. Observed across at least ``min_distinct_days`` different days, OR
2. Observed by at least ``min_distinct_devices`` different devices, OR
3. Explicitly human-verified.

The engine is a pure policy object — it only *decides*; the
:class:`~ai.memory.online_memory.OnlineMemoryManager` performs the actual
copy into permanent memory.
"""

from __future__ import annotations

from ai.memory.observation import Candidate
from ai.utils.logging import get_logger

logger = get_logger("memory.promotion")


class PromotionEngine:
    """Applies the candidate-to-permanent promotion rules."""

    def __init__(self, *, min_distinct_days: int = 5, min_distinct_devices: int = 3) -> None:
        self._min_days = min_distinct_days
        self._min_devices = min_distinct_devices

    def is_eligible(self, candidate: Candidate) -> bool:
        """True if ``candidate`` satisfies at least one promotion rule."""
        if candidate.human_verified:
            return True
        if candidate.distinct_days >= self._min_days:
            return True
        if candidate.distinct_devices >= self._min_devices:
            return True
        return False

    def reason(self, candidate: Candidate) -> str | None:
        """Return which rule qualifies ``candidate`` (for logging), if any."""
        if candidate.human_verified:
            return "human_verified"
        if candidate.distinct_days >= self._min_days:
            return "distinct_days"
        if candidate.distinct_devices >= self._min_devices:
            return "distinct_devices"
        return None

    def select(self, candidates: list[Candidate]) -> list[Candidate]:
        """Return the subset of ``candidates`` eligible for promotion."""
        eligible = [c for c in candidates if self.is_eligible(c)]
        logger.info("promotion_selected", eligible=len(eligible), total=len(candidates))
        return eligible
