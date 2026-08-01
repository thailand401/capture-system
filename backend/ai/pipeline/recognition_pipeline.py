"""End-to-end recognition pipeline.

    image_path
        -> YOLO detection
        -> crop
        -> DINOv2 embedding
        -> FAISS memory search (top-K)
        -> voting
        -> shape / color / blur validation
        -> rule engine
        -> Prediction

Every collaborator is injected, so any stage can be replaced without
touching this orchestration. Stage timings are captured with a
:class:`~ai.utils.benchmark.Benchmark`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ai.cropper.crop_generator import CropGenerator
from ai.detector.detector import Detector
from ai.embedding.embedding_engine import EmbeddingEngine
from ai.matcher.similarity_search import SimilaritySearch
from ai.matcher.voting_engine import VotingEngine
from ai.memory.observation import ObservationContext
from ai.models.embedding import Embedding
from ai.models.prediction import Prediction, TopKMatch
from ai.rule_engine.rule_engine import RuleEngine, RuleInputs
from ai.utils.benchmark import Benchmark
from ai.utils.image_io import image_hash as compute_image_hash
from ai.utils.image_io import load_image
from ai.utils.logging import get_logger
from ai.validator.blur_validator import BlurValidator
from ai.validator.color_validator import ColorValidator
from ai.validator.shape_validator import ShapeValidator

logger = get_logger("pipeline")


@runtime_checkable
class MemoryLearner(Protocol):
    """Online-learning hook the pipeline feeds each prediction into.

    Implemented by :class:`~ai.memory.online_memory.OnlineMemoryManager`; kept
    as a Protocol so the pipeline never depends on the concrete class.
    """

    def observe(
        self,
        *,
        prediction: Prediction,
        embedding: Embedding,
        image_hash: str,
        context: ObservationContext | None = None,
    ): ...


@dataclass(frozen=True, slots=True)
class RecognitionResult:
    """All predictions for one image plus per-stage benchmark timings (ms)."""

    image_path: str
    predictions: list[Prediction] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)


class RecognitionPipeline:
    """Runs detection through rule-engine scoring for a single image."""

    def __init__(
        self,
        *,
        detector: Detector,
        cropper: CropGenerator,
        embedding_engine: EmbeddingEngine,
        similarity_search: SimilaritySearch,
        voting_engine: VotingEngine,
        shape_validator: ShapeValidator,
        color_validator: ColorValidator,
        blur_validator: BlurValidator,
        rule_engine: RuleEngine,
        top_k: int = 20,
        learner: MemoryLearner | None = None,
    ) -> None:
        self._detector = detector
        self._cropper = cropper
        self._embedding_engine = embedding_engine
        self._similarity_search = similarity_search
        self._voting_engine = voting_engine
        self._shape_validator = shape_validator
        self._color_validator = color_validator
        self._blur_validator = blur_validator
        self._rule_engine = rule_engine
        self._top_k = top_k
        self._learner = learner

    def run(self, image_path: str, *, context: ObservationContext | None = None) -> RecognitionResult:
        """Recognize every traffic sign in the image at ``image_path``.

        When a ``learner`` is configured, each prediction is also fed into the
        online-learning hook (subject to its own append gate), carrying the
        optional capture ``context`` (device id / gps / timestamp).
        """
        benchmark = Benchmark()
        logger.info("pipeline_start", image_path=image_path)

        image = load_image(image_path)

        with benchmark.measure("yolo"):
            detections = self._detector.detect(image)

        predictions: list[Prediction] = []
        for detection in detections:
            crop = self._cropper.generate(image, detection.bbox)

            with benchmark.measure("embedding"):
                embedding = self._embedding_engine.encode(crop.image)

            with benchmark.measure("faiss"):
                matches = self._similarity_search.search(embedding, self._top_k)

            vote = self._voting_engine.vote(matches)

            with benchmark.measure("validation"):
                shape = self._shape_validator.validate(crop.image)
                color = self._color_validator.validate(crop.image)
                blur = self._blur_validator.validate(crop.image)

            best_similarity = matches[0].similarity if matches else 0.0
            validation_score = self._rule_engine.evaluate(
                RuleInputs(
                    yolo_confidence=detection.confidence,
                    embedding_similarity=best_similarity,
                    voting_confidence=vote.confidence,
                    shape_score=shape.score,
                    color_score=color.score,
                    blur_score=blur.score,
                )
            )

            prediction = Prediction(
                traffic_sign_id=vote.sign_id,
                similarity=best_similarity,
                validation_score=validation_score,
                bbox=detection.bbox.xyxy,
                crop_path=crop.path,
                yolo_confidence=detection.confidence,
                voting_confidence=vote.confidence,
                shape=shape.shape,
                shape_score=shape.score,
                colors=color.dominant_colors,
                color_score=color.score,
                blur_score=blur.score,
                top_k_matches=[
                    TopKMatch(sign_id=m.sign_id, image_path=m.image_path, similarity=m.similarity)
                    for m in matches
                ],
            )
            predictions.append(prediction)

            if self._learner is not None:
                with benchmark.measure("learning"):
                    self._learner.observe(
                        prediction=prediction,
                        embedding=embedding,
                        image_hash=compute_image_hash(crop.image),
                        context=context,
                    )

        timings = benchmark.as_dict()
        logger.info(
            "pipeline_complete",
            image_path=image_path,
            predictions=len(predictions),
            total_ms=round(timings.get("total", 0.0), 2),
        )
        return RecognitionResult(image_path=image_path, predictions=predictions, timings_ms=timings)
