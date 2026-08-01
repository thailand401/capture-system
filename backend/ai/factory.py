"""Composition root: build a fully-wired recognition pipeline.

This is the *only* place that knows how the concrete implementations fit
together. Everything else depends on interfaces. Swapping DINOv2 for
SigLIP, or FAISS for another index, means changing construction here — no
stage or the pipeline needs to change.

The returned pipeline's memory is ensured ready (loaded if persisted,
otherwise built from the dataset) so callers can immediately run inference.
"""

from __future__ import annotations

from ai.config import EngineConfig
from ai.cropper.crop_generator import CropGenerator
from ai.detector.yolo_detector import YoloTrafficSignDetector
from ai.embedding.dinov2_encoder import DinoV2Encoder
from ai.embedding.embedding_engine import EmbeddingEngine
from ai.embedding.feature_normalizer import FeatureNormalizer
from ai.matcher.similarity_search import SimilaritySearch
from ai.matcher.voting_engine import VotingEngine
from ai.memory.candidate_memory import CandidateMemory
from ai.memory.faiss_index import FaissIndex
from ai.memory.memory_manager import MemoryManager
from ai.memory.memory_optimizer import MemoryOptimizer
from ai.memory.online_memory import OnlineMemoryManager
from ai.memory.promotion_engine import PromotionEngine
from ai.memory.vector_store import VectorStore
from ai.pipeline.recognition_pipeline import RecognitionPipeline
from ai.rule_engine.rule_engine import RuleEngine
from ai.utils.device import resolve_device
from ai.utils.logging import get_logger
from ai.validator.blur_validator import BlurValidator
from ai.validator.color_validator import ColorValidator
from ai.validator.shape_validator import ShapeValidator

logger = get_logger("factory")


def build_memory_manager(config: EngineConfig | None = None) -> MemoryManager:
    """Construct a :class:`MemoryManager` wired with DINOv2 + FAISS."""
    config = config or EngineConfig()
    device = resolve_device(config.device)

    encoder = DinoV2Encoder(model_name=config.embedding_model, device=device)
    engine = EmbeddingEngine(encoder=encoder, normalizer=FeatureNormalizer())
    index = FaissIndex(dim=config.embedding_dim)
    store = VectorStore(embedding_model=config.embedding_model, dim=config.embedding_dim)

    return MemoryManager(
        embedding_engine=engine,
        index=index,
        store=store,
        dataset_dir=config.dataset_dir,
        index_path=config.index_path,
        metadata_path=config.metadata_path,
        image_extensions=config.image_extensions,
    )


def build_online_memory(
    config: EngineConfig | None = None, *, permanent: MemoryManager | None = None
) -> OnlineMemoryManager:
    """Construct the two-layer :class:`OnlineMemoryManager`.

    Reuses an existing permanent :class:`MemoryManager` when provided so the
    pipeline searches and the learning loop share one index instance.
    """
    config = config or EngineConfig()
    permanent = permanent or build_memory_manager(config)

    candidates = CandidateMemory(
        embedding_model=config.embedding_model,
        dim=config.embedding_dim,
        duplicate_threshold=config.online.duplicate_threshold,
        match_threshold=config.online.match_threshold,
    )
    return OnlineMemoryManager(
        permanent=permanent,
        candidates=candidates,
        promotion_engine=PromotionEngine(
            min_distinct_days=config.online.min_distinct_days,
            min_distinct_devices=config.online.min_distinct_devices,
        ),
        optimizer=MemoryOptimizer(redundancy_threshold=config.online.prune_threshold),
        config=config.online,
        memory_dir=config.memory_dir,
        candidates_path=config.candidates_path,
        version_path=config.version_path,
    )


def build_pipeline(config: EngineConfig | None = None) -> RecognitionPipeline:
    """Construct a ready-to-run :class:`RecognitionPipeline`.

    Builds/loads the traffic sign memory, then wires every stage together.
    When online learning is enabled, the pipeline also feeds predictions into
    a shared :class:`OnlineMemoryManager` (candidate memory).
    """
    config = config or EngineConfig()
    device = resolve_device(config.device)
    logger.info("building_pipeline", device=device, dataset=str(config.dataset_dir))

    memory = build_memory_manager(config)
    memory.ensure_ready()

    learner = None
    if config.online.enabled:
        online = build_online_memory(config, permanent=memory)
        online.load_learning_state()  # permanent already ready; restore candidates + version
        learner = online

    detector = YoloTrafficSignDetector(
        model_path=config.yolo_model,
        confidence=config.yolo_confidence,
        iou=config.yolo_iou,
        device=device,
        label=config.detection_label,
    )
    cropper = CropGenerator(output_dir=config.crops_dir, padding=config.crop_padding)

    return RecognitionPipeline(
        detector=detector,
        cropper=cropper,
        embedding_engine=memory.embedding_engine,  # reuse the same encoder instance
        similarity_search=SimilaritySearch(memory=memory, default_k=config.top_k),
        voting_engine=VotingEngine(),
        shape_validator=ShapeValidator(),
        color_validator=ColorValidator(),
        blur_validator=BlurValidator(threshold=config.blur_threshold),
        rule_engine=RuleEngine(weights=config.weights),
        top_k=config.top_k,
        learner=learner,
    )
