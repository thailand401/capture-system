"""Observation + Candidate models for online memory learning.

An :class:`Observation` is a single real-world sighting of a sign (with all
the context needed for promotion decisions). A :class:`Candidate` is a
cluster of near-identical embeddings — repeated sightings of the *same*
physical sign — accumulated in Candidate Memory until it earns promotion to
Permanent Memory.
"""

from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import numpy as np


class ObservationContext(BaseModel):
    """Per-run capture context supplied by the caller (backend/worker)."""

    device_id: str = Field(default="unknown")
    gps: tuple[float, float] | None = Field(default=None, description="(latitude, longitude)")
    timestamp: str | None = Field(default=None, description="ISO-8601; defaults to now (UTC)")


class Observation(BaseModel):
    """One sighting stored against a candidate. Fully JSON-serializable."""

    observation_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    traffic_sign_id: str
    timestamp: str
    device_id: str
    confidence: float
    validation_score: float
    blur_score: float
    image_hash: str
    image_path: str
    gps: tuple[float, float] | None = None

    @property
    def day(self) -> str:
        """The calendar day (``YYYY-MM-DD``) this sighting occurred."""
        return self.timestamp[:10]

    @classmethod
    def from_context(
        cls,
        *,
        traffic_sign_id: str,
        confidence: float,
        validation_score: float,
        blur_score: float,
        image_hash: str,
        image_path: str,
        context: ObservationContext,
    ) -> "Observation":
        """Build an observation, filling in a UTC timestamp when absent."""
        timestamp = context.timestamp or datetime.now(timezone.utc).isoformat()
        return cls(
            traffic_sign_id=traffic_sign_id,
            timestamp=timestamp,
            device_id=context.device_id,
            confidence=confidence,
            validation_score=validation_score,
            blur_score=blur_score,
            image_hash=image_hash,
            image_path=image_path,
            gps=context.gps,
        )


@dataclass
class Candidate:
    """A cluster of near-identical embeddings plus its observation history."""

    candidate_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    observations: list[Observation] = field(default_factory=list)
    vectors: list["np.ndarray"] = field(default_factory=list)
    human_verified: bool = False

    def add(self, observation: Observation, vector: "np.ndarray") -> None:
        """Record a new sighting and its embedding on this candidate."""
        self.observations.append(observation)
        self.vectors.append(vector)

    @property
    def traffic_sign_id(self) -> str:
        """Majority-voted sign id across the candidate's observations."""
        counts = Counter(o.traffic_sign_id for o in self.observations)
        return counts.most_common(1)[0][0]

    @property
    def distinct_days(self) -> int:
        return len({o.day for o in self.observations})

    @property
    def distinct_devices(self) -> int:
        return len({o.device_id for o in self.observations})

    @property
    def observation_count(self) -> int:
        return len(self.observations)

    def best_image_path(self) -> str:
        """Image path of the highest-scoring observation (for provenance)."""
        best = max(self.observations, key=lambda o: o.validation_score)
        return best.image_path

    def representative(self) -> "np.ndarray":
        """L2-normalized mean embedding — the candidate's canonical vector."""
        import numpy as np

        stacked = np.stack(self.vectors).astype(np.float32)
        mean = stacked.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        return (mean / norm).astype(np.float32) if norm > 1e-12 else mean.astype(np.float32)
