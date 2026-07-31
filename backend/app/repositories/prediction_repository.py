"""Data-access layer for ``Prediction`` rows.

Predictions are append-only: this repository intentionally exposes no
update/delete methods, since a capture event may accumulate multiple
predictions over time (e.g. re-processing with a newer model) and old
results must never be overwritten.
"""

from __future__ import annotations

from sqlalchemy import select

from app.models.prediction import Prediction
from app.repositories.base import BaseRepository


class PredictionRepository(BaseRepository):
    """Persistence operations for ``Prediction`` rows."""

    async def create(self, prediction: Prediction) -> Prediction:
        """Persist a new prediction row (never updates an existing one)."""
        self.session.add(prediction)
        await self.session.flush()
        return prediction

    async def list_for_event(self, capture_event_id: int) -> list[Prediction]:
        """Return all predictions for a capture event, newest first.

        Orders by ``created_at`` then ``id`` (both descending): predictions
        inserted within the same transaction can share an identical
        ``created_at`` (e.g. Postgres' ``now()`` is fixed per-transaction),
        so ``id`` breaks ties deterministically.
        """
        stmt = (
            select(Prediction)
            .where(Prediction.capture_event_id == capture_event_id)
            .order_by(Prediction.created_at.desc(), Prediction.id.desc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
