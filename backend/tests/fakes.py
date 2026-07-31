"""Test doubles shared across the test suite.

Keeping these in one module avoids duplicating fake storage/pipeline
implementations across repository, API, and worker tests.
"""

from __future__ import annotations

from app.ai.interfaces import Pipeline
from app.ai.types import PipelineResult
from app.storage.base import StorageBackend


class FakeStorageBackend(StorageBackend):
    """In-memory storage backend used in tests instead of real Supabase Storage."""

    def __init__(self, *, fail_download: bool = False) -> None:
        self.objects: dict[str, bytes] = {}
        self.deleted_paths: list[str] = []
        self.fail_download = fail_download

    async def upload(self, path: str, data: bytes, content_type: str) -> str:
        self.objects[path] = data
        return path

    async def download(self, path: str) -> bytes:
        if self.fail_download:
            raise RuntimeError("Simulated storage download failure.")
        return self.objects[path]

    async def delete(self, path: str) -> None:
        self.objects.pop(path, None)
        self.deleted_paths.append(path)


class FakePipeline(Pipeline):
    """Deterministic AI pipeline stand-in for worker tests."""

    async def run(self, image_path: str) -> PipelineResult:
        return PipelineResult(
            model_name="fake-model",
            model_version="1.0.0",
            traffic_sign_class="stop_sign",
            confidence=0.95,
            ocr_text="STOP",
            validation_score=0.9,
        )
