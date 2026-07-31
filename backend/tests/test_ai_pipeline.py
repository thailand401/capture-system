"""AI pipeline stub tests."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.ai.factory import build_default_pipeline
from app.ai.types import PipelineResult


@pytest.mark.asyncio
async def test_default_pipeline_runs_end_to_end_with_stubs(tmp_path):
    image_path = tmp_path / "sample.jpg"
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), image)

    pipeline = build_default_pipeline()
    result = await pipeline.run(str(image_path))

    assert isinstance(result, PipelineResult)
    assert result.model_name == "yolo"
    # Stub stages report no classification/validation yet.
    assert result.traffic_sign_class is None
    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_default_pipeline_raises_for_missing_image():
    pipeline = build_default_pipeline()
    with pytest.raises(ValueError):
        await pipeline.run("/nonexistent/path/to/image.jpg")
