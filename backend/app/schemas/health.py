"""Health check response schema."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Simple liveness payload."""

    status: str = "ok"
