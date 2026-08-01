"""Engine configuration.

Centralizes every tunable value (paths, model names, thresholds, search
``K`` and rule-engine weights) in one immutable object. Construct it once
and inject it into the factory. Nothing in the engine hardcodes traffic
sign classes — recognition is entirely dataset-driven.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class RuleWeights(BaseModel):
    """Relative weights used by the :class:`~ai.rule_engine.rule_engine.RuleEngine`.

    Weights are normalized to sum to 1.0 at validation time, so callers may
    pass any positive magnitudes and only the *ratios* matter.
    """

    yolo: float = Field(default=0.15, ge=0.0)
    similarity: float = Field(default=0.35, ge=0.0)
    voting: float = Field(default=0.25, ge=0.0)
    shape: float = Field(default=0.10, ge=0.0)
    color: float = Field(default=0.10, ge=0.0)
    blur: float = Field(default=0.05, ge=0.0)

    @model_validator(mode="after")
    def _require_positive_total(self) -> "RuleWeights":
        total = self.yolo + self.similarity + self.voting + self.shape + self.color + self.blur
        if total <= 0:
            raise ValueError("RuleWeights must have a positive total.")
        return self

    def normalized(self) -> dict[str, float]:
        """Return the weights as a dict scaled so the values sum to 1.0."""
        raw = {
            "yolo": self.yolo,
            "similarity": self.similarity,
            "voting": self.voting,
            "shape": self.shape,
            "color": self.color,
            "blur": self.blur,
        }
        total = sum(raw.values())
        return {key: value / total for key, value in raw.items()}


class OnlineLearningConfig(BaseModel):
    """Thresholds governing the two-layer online memory learning loop."""

    enabled: bool = Field(default=True, description="Feed predictions back into candidate memory")

    # --- Candidate append gate (pipeline -> candidate memory) ---
    min_validation_score: float = Field(default=90.0, ge=0.0, le=100.0)
    min_similarity: float = Field(default=0.95, ge=0.0, le=1.0)

    # --- Duplicate + clustering thresholds ---
    duplicate_threshold: float = Field(default=0.995, ge=0.0, le=1.0, description="Above this = ignore")
    match_threshold: float = Field(default=0.95, ge=0.0, le=1.0, description="Above this = same candidate")

    # --- Promotion rules (candidate -> permanent) ---
    min_distinct_days: int = Field(default=5, gt=0)
    min_distinct_devices: int = Field(default=3, gt=0)

    # --- Pruning (Memory Optimizer) ---
    prune_threshold: float = Field(default=0.98, ge=0.0, le=1.0, description="Above this = redundant")


class EngineConfig(BaseModel):
    """Immutable configuration for the whole recognition engine."""

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    # --- Filesystem layout ---
    dataset_dir: Path = Field(default=Path("dataset"), description="Root of per-sign template folders")
    memory_dir: Path = Field(default=Path("memory"), description="Where FAISS index + metadata live")
    crops_dir: Path = Field(default=Path("crops"), description="Where generated crops are written")

    # --- Detection (YOLO) ---
    yolo_model: str = Field(default="yolov8n.pt", description="Ultralytics weights path or name")
    yolo_confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    yolo_iou: float = Field(default=0.45, ge=0.0, le=1.0)
    detection_label: str = Field(default="traffic_sign", description="Single class label for all detections")
    crop_padding: float = Field(default=0.05, ge=0.0, description="Fractional padding added around a box")

    # --- Embedding (DINOv2) ---
    embedding_model: str = Field(default="facebook/dinov2-base")
    embedding_dim: int = Field(default=768, gt=0)

    # --- Compute backend ---
    device: str | None = Field(default=None, description="cuda|mps|cpu; None = auto-detect")

    # --- Memory / search ---
    top_k: int = Field(default=20, gt=0, description="Number of nearest neighbours retrieved")
    image_extensions: tuple[str, ...] = Field(default=(".png", ".jpg", ".jpeg", ".bmp", ".webp"))

    # --- Validators ---
    blur_threshold: float = Field(default=150.0, gt=0.0, description="Laplacian variance = sharp")

    # --- Rule engine ---
    weights: RuleWeights = Field(default_factory=RuleWeights)

    # --- Online memory learning ---
    online: OnlineLearningConfig = Field(default_factory=OnlineLearningConfig)

    @property
    def index_path(self) -> Path:
        """Absolute-ish path of the persisted FAISS index."""
        return self.memory_dir / "vectors.faiss"

    @property
    def metadata_path(self) -> Path:
        """Path of the persisted memory metadata JSON."""
        return self.memory_dir / "metadata.json"

    @property
    def candidates_path(self) -> Path:
        """Path of the persisted candidate memory JSON."""
        return self.memory_dir / "candidates.json"

    @property
    def version_path(self) -> Path:
        """Path of the memory version/history JSON."""
        return self.memory_dir / "version.json"
