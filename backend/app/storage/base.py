"""Abstract storage backend interface.

Defining this as an interface (rather than importing the Supabase client
directly throughout the codebase) means the temporary-image backend could
later be swapped (e.g. for S3 or GCS) without touching services, routes,
or the worker.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class StorageBackend(ABC):
    """Contract for a temporary object storage backend."""

    @abstractmethod
    async def upload(self, path: str, data: bytes, content_type: str) -> str:
        """Upload bytes to ``path`` and return the stored path."""

    @abstractmethod
    async def download(self, path: str) -> bytes:
        """Download and return the raw bytes stored at ``path``."""

    @abstractmethod
    async def delete(self, path: str) -> None:
        """Delete the object stored at ``path``. Must be idempotent."""
