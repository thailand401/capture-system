"""Lightweight stage benchmarking utility.

Accumulates wall-clock time per named stage so the pipeline can report
YOLO / embedding / FAISS / validation / total timings without pulling in a
profiler.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter


class Benchmark:
    """Accumulates elapsed milliseconds per named stage."""

    def __init__(self) -> None:
        self._timings_ms: dict[str, float] = {}

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        """Context manager that adds elapsed time to ``stage``'s total."""
        start = perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (perf_counter() - start) * 1000.0
            self._timings_ms[stage] = self._timings_ms.get(stage, 0.0) + elapsed_ms

    def record(self, stage: str, elapsed_ms: float) -> None:
        """Manually add ``elapsed_ms`` to ``stage``'s total."""
        self._timings_ms[stage] = self._timings_ms.get(stage, 0.0) + elapsed_ms

    @property
    def total_ms(self) -> float:
        return sum(self._timings_ms.values())

    def as_dict(self) -> dict[str, float]:
        """Return a copy of per-stage timings plus a ``total`` entry (ms)."""
        result = dict(self._timings_ms)
        result["total"] = self.total_ms
        return result
