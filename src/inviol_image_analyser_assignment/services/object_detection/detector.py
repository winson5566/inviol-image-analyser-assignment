"""Shared object-detector contract and adapter selection."""

from typing import Protocol

from PIL import Image

from inviol_image_analyser_assignment.config import DetectorType, ObjectDetectionConfig
from inviol_image_analyser_assignment.models import ObjectDetectionResult
from inviol_image_analyser_assignment.services.object_detection.grounding_dino import GroundingDinoDetector
from inviol_image_analyser_assignment.services.object_detection.yolo_world import YoloWorldDetector


class ObjectDetector(Protocol):
    """Interface consumed by the analysis pipeline and test doubles."""

    def detect(self, image: Image.Image) -> ObjectDetectionResult:
        """Detect configured workplace objects in an image."""

        ...


def create_object_detector(config: ObjectDetectionConfig) -> ObjectDetector:
    """Create the adapter selected by the detector release configuration."""

    if config.detector_type is DetectorType.YOLO_WORLD:
        return YoloWorldDetector(config)
    if config.detector_type is DetectorType.GROUNDING_DINO:
        return GroundingDinoDetector(config)
    raise ValueError(f"Unsupported detector type: {config.detector_type}")
