"""Shared enumerations used across ORM models and schemas."""

from __future__ import annotations

import enum


class CaptureStatus(enum.StrEnum):
    """Lifecycle status of a capture event, from ingestion to completion."""

    NEW = "NEW"
    DOWNLOADING = "DOWNLOADING"
    PROCESSING = "PROCESSING"
    DONE = "DONE"
    REJECTED = "REJECTED"
    ERROR = "ERROR"
