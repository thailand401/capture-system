"""End-to-end pipeline unit test wired entirely with fakes.

Exercises detection -> crop -> embedding -> memory search -> voting ->
validation -> rule engine -> Prediction using numpy-only doubles, so no
torch/faiss/ultralytics/OpenCV is required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ai.cropper import crop_generator as crop_generator_module
from ai.cropper.crop_generator import CropGenerator
from ai.matcher.similarity_search import SimilaritySearch
from ai.matcher.voting_engine import VotingEngine
from ai.memory import memory_manager as memory_manager_module
from ai.pipeline import recognition_pipeline as pipeline_module
from ai.pipeline.recognition_pipeline import RecognitionPipeline
from ai.rule_engine.rule_engine import RuleEngine
from tests_ai.fakes import (
    FakeBlurValidator,
    FakeColorValidator,
    FakeDetector,
    FakeShapeValidator,
)


def _color(sign_id: str) -> np.ndarray:
    value = (abs(hash(sign_id)) % 200) + 30
    return np.full((16, 16, 3), value, dtype=np.uint8)


@pytest.fixture
def pipeline(memory, monkeypatch):
    monkeypatch.setattr(memory_manager_module, "load_image", lambda p: _color(str(p).split("/")[-2]))
    for sign_id in ("102", "103", "104"):
        memory.append(image_path=f"dataset/{sign_id}/0.png", sign_id=sign_id, persist=False)

    # Feed the pipeline an input image identical to sign "102".
    monkeypatch.setattr(pipeline_module, "load_image", lambda p: _color("102"))
    monkeypatch.setattr(crop_generator_module, "save_image", lambda img, path: Path(path))

    return RecognitionPipeline(
        detector=FakeDetector(confidence=0.9),
        cropper=CropGenerator(output_dir="crops", padding=0.0),
        embedding_engine=memory.embedding_engine,
        similarity_search=SimilaritySearch(memory=memory, default_k=3),
        voting_engine=VotingEngine(),
        shape_validator=FakeShapeValidator(),
        color_validator=FakeColorValidator(),
        blur_validator=FakeBlurValidator(),
        rule_engine=RuleEngine(),
        top_k=3,
    )


def test_pipeline_recognizes_closest_sign(pipeline):
    result = pipeline.run("input.png")
    assert len(result.predictions) == 1
    prediction = result.predictions[0]
    assert prediction.traffic_sign_id == "102"
    assert 0.0 <= prediction.validation_score <= 100.0
    assert prediction.top_k_matches
    assert prediction.yolo_confidence == 0.9


def test_pipeline_reports_stage_timings(pipeline):
    result = pipeline.run("input.png")
    for stage in ("yolo", "embedding", "faiss", "validation", "total"):
        assert stage in result.timings_ms
