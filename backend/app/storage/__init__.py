"""Temporary image storage backends (Supabase Storage by default)."""

from app.storage.base import StorageBackend
from app.storage.supabase_storage import SupabaseStorageService

__all__ = ["StorageBackend", "SupabaseStorageService"]
