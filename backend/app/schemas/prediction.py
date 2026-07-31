"""Pydantic schemas for AI prediction results."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class PredictionResponse(BaseModel):
    """A single AI pipeline prediction result attached to a capture event."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    model_name: str
    model_version: str
    traffic_sign_class: str | None
    confidence: float | None
    ocr_text: str | None
    validation_score: float | None
    created_at: datetime
