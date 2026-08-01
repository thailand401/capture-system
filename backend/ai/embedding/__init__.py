"""Embedding package: image -> L2-normalized feature vector."""

from __future__ import annotations

from ai.embedding.dinov2_encoder import DinoV2Encoder
from ai.embedding.embedding_engine import EmbeddingEngine, ImageEncoder
from ai.embedding.feature_normalizer import FeatureNormalizer

__all__ = ["DinoV2Encoder", "EmbeddingEngine", "FeatureNormalizer", "ImageEncoder"]
