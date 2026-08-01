"""Matcher package: similarity search + voting over memory hits."""

from __future__ import annotations

from ai.matcher.similarity_search import SimilaritySearch
from ai.matcher.voting_engine import VoteResult, VotingEngine

__all__ = ["SimilaritySearch", "VoteResult", "VotingEngine"]
