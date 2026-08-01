"""YOLO detector unit tests using a fake Ultralytics model.

Avoids importing torch/ultralytics by injecting a fake model into the
already-lazy detector, so only the box-extraction logic is exercised.
"""

from __future__ import annotations

import numpy as np

from ai.detector.yolo_detector import YoloTrafficSignDetector


class _FakeTensorRow:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


class _FakeScalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _FakeBox:
    def __init__(self, xyxy, conf):
        self.xyxy = [_FakeTensorRow(xyxy)]
        self.conf = [_FakeScalar(conf)]


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    def __init__(self, boxes):
        self._boxes = boxes

    def predict(self, **_kwargs):
        return [_FakeResult(self._boxes)]


def _detector_with(boxes) -> YoloTrafficSignDetector:
    detector = YoloTrafficSignDetector(model_path="unused.pt")
    detector._model = _FakeModel(boxes)  # noqa: SLF001 - inject fake model
    detector._device = "cpu"  # noqa: SLF001
    return detector


def test_detect_extracts_boxes_as_traffic_sign():
    detector = _detector_with([_FakeBox([10.0, 20.0, 50.0, 80.0], 0.87)])
    detections = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
    assert len(detections) == 1
    det = detections[0]
    assert det.label == "traffic_sign"
    assert det.confidence == 0.87
    assert det.bbox.to_int() == (10, 20, 50, 80)


def test_detect_handles_no_boxes():
    detector = _detector_with([])
    assert detector.detect(np.zeros((10, 10, 3), dtype=np.uint8)) == []


def test_detect_returns_multiple():
    detector = _detector_with(
        [_FakeBox([0, 0, 5, 5], 0.9), _FakeBox([6, 6, 9, 9], 0.6)]
    )
    assert len(detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))) == 2
