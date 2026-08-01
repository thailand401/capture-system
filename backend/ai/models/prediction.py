"""Final prediction object returned by the recognition pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TopKMatch(BaseModel):
    """One nearest-neighbour hit from the traffic sign memory."""

    sign_id: str
    image_path: str
    similarity: float


class Prediction(BaseModel):
    """The engine's verdict for a single detected traffic sign.

    Fully JSON-serializable (no numpy) so the backend can persist or return
    it directly.
    """

    traffic_sign_id: str | None = Field(default=None, description="Voted sign id, or None if unknown")
    similarity: float = Field(default=0.0, description="Best cosine similarity to the memory")
    validation_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Aggregate 0..100 score")
    bbox: tuple[float, float, float, float] = Field(description="Detection box in xyxy pixels")
    crop_path: str | None = Field(default=None, description="Where the generated crop was saved")

    yolo_confidence: float = Field(default=0.0)
    voting_confidence: float = Field(default=0.0)

    shape: str | None = Field(default=None)
    shape_score: float = Field(default=0.0)
    colors: list[str] = Field(default_factory=list)
    color_score: float = Field(default=0.0)
    blur_score: float = Field(default=0.0)

    top_k_matches: list[TopKMatch] = Field(default_factory=list)
