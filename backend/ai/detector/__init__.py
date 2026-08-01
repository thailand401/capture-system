"""Detector package: locate traffic-sign regions in a full image."""

from __future__ import annotations

from ai.detector.detector import Detection, Detector
from ai.detector.yolo_detector import YoloTrafficSignDetector

__all__ = ["Detection", "Detector", "YoloTrafficSignDetector"]
