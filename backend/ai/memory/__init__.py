"""Persistent traffic sign memory + online learning.

Two layers:
- **Permanent Memory** (:class:`MemoryManager`) — verified vectors used for
  recognition search (the source of truth).
- **Candidate Memory** (:class:`CandidateMemory`) — unverified discoveries
  awaiting promotion.

:class:`OnlineMemoryManager` coordinates both, plus promotion, pruning and
versioning.
"""

from __future__ import annotations

from ai.memory.candidate_memory import AppendResult, AppendStatus, CandidateMemory
from ai.memory.faiss_index import FaissIndex, VectorIndex
from ai.memory.memory_manager import MemoryManager, MemoryMatch
from ai.memory.memory_optimizer import MemoryOptimizer
from ai.memory.observation import Candidate, Observation, ObservationContext
from ai.memory.online_memory import MemoryStatistics, OnlineMemoryManager, PromotionResult
from ai.memory.promotion_engine import PromotionEngine
from ai.memory.vector_store import MemoryEntry, VectorStore

__all__ = [
    "AppendResult",
    "AppendStatus",
    "Candidate",
    "CandidateMemory",
    "FaissIndex",
    "MemoryEntry",
    "MemoryManager",
    "MemoryMatch",
    "MemoryOptimizer",
    "MemoryStatistics",
    "Observation",
    "ObservationContext",
    "OnlineMemoryManager",
    "PromotionEngine",
    "PromotionResult",
    "VectorIndex",
    "VectorStore",
]
