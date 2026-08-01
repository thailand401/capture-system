"""Structured logging via Loguru.

The engine logs one structured record per stage. ``get_logger`` returns a
Loguru logger bound with a ``component`` field so downstream sinks can
filter/aggregate by pipeline stage.
"""

from __future__ import annotations

import sys
from typing import Any

from loguru import logger

_CONFIGURED = False


def configure_logging(*, level: str = "INFO", serialize: bool = False) -> None:
    """Configure the global Loguru sink once.

    Args:
        level: Minimum log level.
        serialize: Emit JSON records when True (production), pretty console
            output otherwise (development).
    """
    global _CONFIGURED
    logger.remove()
    logger.add(
        sys.stdout,
        level=level.upper(),
        serialize=serialize,
        backtrace=False,
        diagnose=False,
        enqueue=False,
    )
    _CONFIGURED = True


def get_logger(component: str, **context: Any):
    """Return a Loguru logger bound to ``component`` (plus optional context)."""
    if not _CONFIGURED:
        configure_logging()
    return logger.bind(component=component, **context)
