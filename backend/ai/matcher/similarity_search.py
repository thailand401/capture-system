"""Similarity search: thin query facade over the memory manager.

Keeps the pipeline decoupled from the memory's internals and owns the
default ``K``. Replaceable independently (e.g. to add re-ranking) without
touching the pipeline.
"""

from __future__ import annotations

from ai.memory.memory_manager import MemoryManager, MemoryMatch
from ai.models.embedding import Embedding
from ai.utils.logging import get_logger

logger = get_logger("matcher.similarity")


class SimilaritySearch:
    """Retrieve the top-``K`` nearest memory entries for an embedding."""

    def __init__(self, *, memory: MemoryManager, default_k: int = 20) -> None:
        self._memory = memory
        self._default_k = default_k

    def search(self, embedding: Embedding, k: int | None = None) -> list[MemoryMatch]:
        """Return the top-``k`` (or default ``K``) matches for ``embedding``."""
        top_k = k or self._default_k
        matches = self._memory.search(embedding, top_k)
        logger.info("similarity_search", k=top_k, returned=len(matches))
        return matches
