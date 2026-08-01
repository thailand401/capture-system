"""Embedding engine: orchestrates an encoder + normalizer.

The encoder is an injectable interface (:class:`ImageEncoder`) so DINOv2,
SigLIP, or a test fake can be used interchangeably. The engine's only job is
to run the encoder and L2-normalize its output into reusable
:class:`~ai.models.embedding.Embedding` value objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ai.embedding.feature_normalizer import FeatureNormalizer
from ai.models.embedding import Embedding
from ai.utils.logging import get_logger

if TYPE_CHECKING:
    import numpy as np

logger = get_logger("embedding.engine")


class ImageEncoder(ABC):
    """Turns images into raw (un-normalized) feature vectors."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the underlying model (stored in memory metadata)."""

    @abstractmethod
    def encode(self, images: list["np.ndarray"]) -> "np.ndarray":
        """Return a ``(n, dim)`` float32 array of raw embeddings for ``images``."""


class EmbeddingEngine:
    """Encodes images and returns L2-normalized, reusable embeddings."""

    def __init__(self, *, encoder: ImageEncoder, normalizer: FeatureNormalizer | None = None) -> None:
        self._encoder = encoder
        self._normalizer = normalizer or FeatureNormalizer()

    @property
    def model_name(self) -> str:
        return self._encoder.model_name

    def encode(self, image: "np.ndarray") -> Embedding:
        """Encode a single image into a normalized :class:`Embedding`."""
        matrix = self.encode_batch([image])
        return Embedding(vector=matrix[0], model_name=self._encoder.model_name)

    def encode_batch(self, images: list["np.ndarray"]) -> "np.ndarray":
        """Encode a batch and return a ``(n, dim)`` normalized float32 array."""
        raw = self._encoder.encode(images)
        normalized = self._normalizer.normalize(raw)
        logger.info("embeddings_encoded", count=len(images), dim=int(normalized.shape[-1]) if len(images) else 0)
        return normalized
