"""Health check endpoint for uptime probes / container orchestration."""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Lightweight liveness check. Does not touch the database or storage."""
    return HealthResponse()
