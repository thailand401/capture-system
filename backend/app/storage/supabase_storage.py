"""Supabase Storage backend implementation.

Wraps the official ``supabase-py`` async client to upload, download and
delete objects in the temporary image bucket. All operations are retried
with exponential backoff (network calls to Supabase can transiently fail)
and every attempt is structurally logged per the project's audit
requirements.
"""

from __future__ import annotations

from supabase import AsyncClient, acreate_client

from app.core.config import get_settings
from app.core.exceptions import StorageError
from app.core.logging import get_logger
from app.storage.base import StorageBackend
from app.utils.retry import default_retry

logger = get_logger(__name__)


class SupabaseStorageService(StorageBackend):
    """Temporary-image storage backend backed by a Supabase Storage bucket."""

    def __init__(self, bucket: str | None = None) -> None:
        settings = get_settings()
        self._bucket = bucket or settings.supabase_bucket
        self._supabase_url = settings.supabase_url
        self._supabase_key = settings.supabase_key
        self._client: AsyncClient | None = None

    async def _get_client(self) -> AsyncClient:
        """Lazily create (and cache) the Supabase async client."""
        if self._client is None:
            self._client = await acreate_client(self._supabase_url, self._supabase_key)
        return self._client

    @default_retry()
    async def upload(self, path: str, data: bytes, content_type: str) -> str:
        """Upload ``data`` to ``path`` inside the bucket, overwriting if present."""
        client = await self._get_client()
        try:
            await client.storage.from_(self._bucket).upload(
                path=path,
                file=data,
                file_options={"content-type": content_type, "upsert": "true"},
            )
        except Exception as exc:
            logger.error("storage_upload_failed", path=path, bucket=self._bucket)
            raise StorageError(f"Failed to upload '{path}' to bucket '{self._bucket}'.") from exc
        logger.info("storage_upload_succeeded", path=path, bucket=self._bucket, bytes=len(data))
        return path

    @default_retry()
    async def download(self, path: str) -> bytes:
        """Download and return the raw bytes stored at ``path``."""
        client = await self._get_client()
        try:
            data: bytes = await client.storage.from_(self._bucket).download(path)
        except Exception as exc:
            logger.error("storage_download_failed", path=path, bucket=self._bucket)
            raise StorageError(f"Failed to download '{path}' from bucket '{self._bucket}'.") from exc
        logger.info("storage_download_succeeded", path=path, bucket=self._bucket, bytes=len(data))
        return data

    @default_retry()
    async def delete(self, path: str) -> None:
        """Delete the object at ``path``. Idempotent: missing objects are not an error."""
        client = await self._get_client()
        try:
            await client.storage.from_(self._bucket).remove([path])
        except Exception as exc:
            logger.error("storage_delete_failed", path=path, bucket=self._bucket)
            raise StorageError(f"Failed to delete '{path}' from bucket '{self._bucket}'.") from exc
        logger.info("storage_delete_succeeded", path=path, bucket=self._bucket)
