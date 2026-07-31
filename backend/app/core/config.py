"""Centralized application configuration.

All configuration is sourced from environment variables (optionally via a
local ``.env`` file for development). Never hardcode secrets or
environment-specific values elsewhere in the codebase — always go through
``Settings`` / ``get_settings()``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- App metadata ---
    app_name: str = "capture-system-backend"
    environment: str = "development"
    debug: bool = False

    # --- Database ---
    database_url: str = Field(..., description="Async SQLAlchemy URL, e.g. postgresql+asyncpg://...")

    # --- Supabase ---
    # SUPABASE_KEY must be the *service role* key (server-side only) so the
    # backend can manage the temporary image bucket. It must never be
    # exposed to the Android app or any other client.
    supabase_url: str = Field(..., description="Supabase project URL")
    supabase_key: str = Field(..., description="Supabase service role key (server-side only)")
    supabase_bucket: str = Field(
        default="traffic-sign-temp", description="Storage bucket for temporary images"
    )

    # --- Local cache ---
    local_cache: Path = Field(
        default=Path("./cache"), description="Local dir for images downloaded by the worker"
    )

    # --- AI ---
    yolo_model: str = Field(default="yolov8n.pt", description="Path or name of the YOLO model weights")

    # --- Background worker ---
    worker_enabled: bool = Field(
        default=True, description="Whether the background worker starts with the app"
    )
    worker_interval_seconds: float = Field(default=5.0, description="Polling interval between worker cycles")
    worker_batch_size: int = Field(default=1, description="Number of NEW events claimed per worker cycle")
    worker_max_retries: int = Field(default=3, description="Max retry attempts for storage operations")
    worker_retry_backoff_seconds: float = Field(
        default=2.0, description="Base backoff seconds between retries"
    )

    # --- Uploads ---
    max_upload_size_bytes: int = Field(
        default=15 * 1024 * 1024, description="Max size (bytes) for an uploaded image/thumbnail"
    )

    # --- Pagination ---
    default_page_size: int = 20
    max_page_size: int = 100

    # --- Logging ---
    log_level: str = "INFO"
    log_json: bool = True


@lru_cache
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance (singleton for process lifetime)."""
    return Settings()  # type: ignore[call-arg]
