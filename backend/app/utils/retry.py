"""Reusable retry policy for flaky I/O operations.

Centralizes the retry/backoff strategy used for Supabase Storage calls so
every caller shares the same, configurable policy instead of hand-rolling
retry loops.
"""

from __future__ import annotations

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import get_settings


def default_retry():
    """Build the standard retry decorator from the current worker settings.

    Retries any exception up to ``WORKER_MAX_RETRIES`` attempts with
    exponential backoff starting at ``WORKER_RETRY_BACKOFF_SECONDS``. The
    final exception is always re-raised (``reraise=True``) so callers can
    still handle/log the failure after retries are exhausted.
    """
    settings = get_settings()
    return retry(
        reraise=True,
        stop=stop_after_attempt(settings.worker_max_retries),
        wait=wait_exponential(multiplier=settings.worker_retry_backoff_seconds, min=1, max=30),
        retry=retry_if_exception_type(Exception),
    )
