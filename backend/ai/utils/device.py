"""Compute backend detection for Apple Silicon / CUDA / CPU.

Torch is imported lazily so importing this module (and the engine) does not
require torch until inference actually runs.
"""

from __future__ import annotations

from ai.utils.logging import get_logger

logger = get_logger("device")


def resolve_device(preferred: str | None = None) -> str:
    """Return the best available torch device string.

    Priority: explicit ``preferred`` -> CUDA -> Apple MPS -> CPU. Falls back
    gracefully if the requested device is unavailable.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is an optional heavy dep
        logger.warning("torch_unavailable_defaulting_cpu")
        return "cpu"

    if preferred:
        preferred = preferred.lower()
        if preferred == "cuda" and torch.cuda.is_available():
            return "cuda"
        if preferred == "mps" and torch.backends.mps.is_available():
            return "mps"
        if preferred == "cpu":
            return "cpu"
        logger.warning("preferred_device_unavailable", preferred=preferred)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info("device_selected", device=device)
    return device
