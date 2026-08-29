"""Tests for typed detection configuration and the YOLO-World adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Self, final

from PIL import Image

from inviol_image_analyser_assignment.config import (
    DetectionTargetConfig,
    DetectorType,
    ObjectDetectionConfig,
    load_analysis_config,
)
from inviol_image_analyser_assignment.models import ObjectType
from inviol_image_analyser_assignment.services.object_detection import (
    YoloWorldDetector,
    create_object_detector,
)


@final
class FakeTensor[T]:
    """Small tensor-like value supporting the conversion chain used by the adapter."""

    def __init__(self, values: T) -> None:
        self._values: T = values

    def detach(self) -> Self:
        return self

    def cpu(self) -> Self:
        return self

    def tolist(self) -> T:
        return self._values


@final
class FakeBoxes:
    def __init__(self, coordinates: list[list[float]], class_ids: list[float], confidences: list[float]) -> None:
        self.xyxyn: FakeTensor[list[list[float]]] = FakeTensor(coordinates)
        self.cls: FakeTensor[list[float]] = FakeTensor(class_ids)
        self.conf: FakeTensor[list[float]] = FakeTensor(confidences)


@final
class FakeResult:
    def __init__(self, boxes: FakeBoxes | None) -> None:
        self.boxes: FakeBoxes | None = boxes
        self.names: list[str] = ["person", "material handling vehicle", "safety hat", "safety vest", "unknown"]


@final
class FakeYoloModel:
    def __init__(self, results: list[FakeResult]) -> None:
        self.results: list[FakeResult] = results
        self.classes: list[str] = []
        self.last_confidence: float | None = None

    def set_classes(self, classes: list[str]) -> None:
        self.classes = classes

    def predict(
        self,
        *,
        source: Image.Image,
        conf: float,
        iou: float,
        imgsz: int,
        device: str,
        verbose: bool,
    ) -> list[FakeResult]:
        del source, iou, imgsz, device, verbose
        self.last_confidence = conf
        return self.results


def detection_config() -> ObjectDetectionConfig:
    return ObjectDetectionConfig(
        detector_type=DetectorType.YOLO_WORLD,
        model_source="test-model.pt",
        image_size=640,
        nms_iou_threshold=0.5,
        targets={
            ObjectType.PERSON: DetectionTargetConfig(prompt="person", confidence_threshold=0.3),
            ObjectType.FORKLIFT: DetectionTargetConfig(prompt="material handling vehicle", confidence_threshold=0.035),
            ObjectType.SAFETY_HAT: DetectionTargetConfig(prompt="safety hat", confidence_threshold=0.035),
            ObjectType.SAFETY_VEST: DetectionTargetConfig(prompt="safety vest", confidence_threshold=0.05),
        },
    )


def test_loads_active_detection_config() -> None:
    config_path = Path(__file__).parents[1] / "config" / "analysis.json"

    config = load_analysis_config(config_path)

    assert config.object_detection.detector_type is DetectorType.YOLO_WORLD
    assert config.object_detection.device == "cpu"
    assert config.object_detection.model_source == "weights/yolo_world/workplace-safety-yolo-world-v0.1.pt"
    assert config.object_detection.targets[ObjectType.PERSON].confidence_threshold == 0.04
    assert config.object_detection.targets[ObjectType.FORKLIFT].confidence_threshold == 0.04
    assert isinstance(create_object_detector(config.object_detection), YoloWorldDetector)


def test_loads_configured_model_and_handles_empty_result() -> None:
    fake_model = FakeYoloModel([FakeResult(None)])
    loaded_sources: list[str] = []

    def model_factory(source: str) -> FakeYoloModel:
        loaded_sources.append(source)
        return fake_model

    detector = YoloWorldDetector(detection_config(), model_factory=model_factory)

    result = detector.detect(Image.new("RGB", (10, 10)))

    assert loaded_sources == ["test-model.pt"]
    assert result.image.model_dump() == {"width": 10, "height": 10}
    assert result.detections == []


def test_converts_and_filters_detections_by_class_confidence() -> None:
    fake_model = FakeYoloModel(
        [
            FakeResult(
                FakeBoxes(
                    coordinates=[
                        [0.1, 0.1, 0.4, 0.9],
                        [-0.1, 0.2, 1.1, 0.8],
                        [0.2, 0.3, 0.5, 0.7],
                        [0.1, 0.1, 0.2, 0.2],
                    ],
                    class_ids=[0.0, 1.0, 3.0, 4.0],
                    confidences=[0.29, 0.2, 0.06, 0.99],
                )
            )
        ]
    )
    loaded_sources: list[str] = []

    def model_factory(source: str) -> FakeYoloModel:
        loaded_sources.append(source)
        return fake_model

    detector = YoloWorldDetector(detection_config(), model_factory=model_factory)

    result = detector.detect(Image.new("RGB", (100, 100)))
    detector.detect(Image.new("RGB", (100, 100)))

    assert loaded_sources == ["test-model.pt"]
    assert fake_model.classes == ["person", "material handling vehicle", "safety hat", "safety vest"]
    assert fake_model.last_confidence == 0.035
    assert result.image.model_dump() == {"width": 100, "height": 100}
    assert [detection.object_type for detection in result.detections] == [
        ObjectType.FORKLIFT,
        ObjectType.SAFETY_VEST,
    ]
    assert result.detections[0].confidence == 0.2
    assert result.detections[0].bounding_box.model_dump() == {
        "x_min": 0.0,
        "y_min": 0.2,
        "x_max": 1.0,
        "y_max": 0.8,
    }
