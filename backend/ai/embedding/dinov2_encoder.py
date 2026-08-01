"""DINOv2 image encoder (HuggingFace Transformers).

Produces a raw (un-normalized) feature vector per image using the CLS token
of ``facebook/dinov2-base`` (768-d by default). Model + processor are loaded
lazily so the module imports cleanly without torch/transformers installed.

Swap this class for a SigLIP encoder by implementing the same
:class:`~ai.embedding.embedding_engine.ImageEncoder` interface — nothing else
in the engine changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ai.embedding.embedding_engine import ImageEncoder
from ai.utils.device import resolve_device
from ai.utils.logging import get_logger

if TYPE_CHECKING:
    import numpy as np

logger = get_logger("embedding.dinov2")


class DinoV2Encoder(ImageEncoder):
    """Encode BGR images into DINOv2 CLS-token feature vectors."""

    def __init__(self, *, model_name: str = "facebook/dinov2-base", device: str | None = None) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any | None = None
        self._processor: Any | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self._device = resolve_device(self._device)
        logger.info("loading_encoder", model=self._model_name, device=self._device)
        self._processor = AutoImageProcessor.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name).to(self._device)
        self._model.eval()
        self._torch = torch

    def encode(self, images: list["np.ndarray"]) -> "np.ndarray":
        """Return a ``(len(images), dim)`` float32 array of raw CLS embeddings."""
        import cv2
        import numpy as np

        if not images:
            return np.empty((0, 0), dtype=np.float32)

        self._ensure_model()
        assert self._processor is not None and self._model is not None
        torch = self._torch

        # Transformers image processors expect RGB; OpenCV gives BGR.
        rgb_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
        inputs = self._processor(images=rgb_images, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        # CLS token = first position of the last hidden state.
        cls = outputs.last_hidden_state[:, 0, :]
        return cls.detach().to("cpu").numpy().astype(np.float32)
