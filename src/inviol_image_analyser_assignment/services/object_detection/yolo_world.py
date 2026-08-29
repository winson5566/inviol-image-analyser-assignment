"""YOLO-World object-detector adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from threading import Lock
from time import perf_counter_ns
from typing import Protocol, Self, cast, final

from PIL import Image

from inviol_image_analyser_assignment.config import ObjectDetectionConfig
from inviol_image_analyser_assignment.models import Detection, ImageDimensions, ObjectDetectionResult, ObjectType
from inviol_image_analyser_assignment.services.object_detection.geometry import normalized_bounding_box


class TensorOutput[T](Protocol):
    """Tensor operations used when converting an Ultralytics result."""

    def detach(self) -> Self: ...

    def cpu(self) -> Self: ...

    def tolist(self) -> T: ...


class PredictionBoxes(Protocol):
    """Subset of Ultralytics Boxes consumed by this adapter."""

    xyxyn: TensorOutput[list[list[float]]]
    cls: TensorOutput[list[float]]
    conf: TensorOutput[list[float]]


class PredictionResult(Protocol):
    """Subset of an Ultralytics Result consumed by this adapter."""

    boxes: PredictionBoxes | None
    names: Sequence[str] | Mapping[int, str]


class YoloModel(Protocol):
    """Operations required from the third-party YOLO model."""

    def set_classes(self, classes: list[str]) -> None: ...

    def predict(
        self,
        *,
        source: Image.Image,
        conf: float,
        iou: float,
        imgsz: int,
        device: str,
        verbose: bool,
    ) -> Sequence[PredictionResult]: ...


ModelFactory = Callable[[str], object]


@final
class YoloWorldDetector:
    """Detect prompted workplace objects with a lazily loaded YOLO-World model."""

    def __init__(self, config: ObjectDetectionConfig, model_factory: ModelFactory | None = None) -> None:
        self._config: ObjectDetectionConfig = config
        self._model_factory: ModelFactory = model_factory or _create_yolo_model
        self._model: YoloModel | None = None
        self._model_lock: Lock = Lock()
        self._object_type_by_prompt: dict[str, ObjectType] = {
            target.prompt.casefold(): object_type for object_type, target in config.targets.items()
        }

    def detect(self, image: Image.Image) -> ObjectDetectionResult:
        """Run inference and return normalized detections above their class thresholds."""

        started_at = perf_counter_ns()
        model = self._get_model()
        results = model.predict(
            source=image,
            conf=min(target.confidence_threshold for target in self._config.targets.values()),
            iou=self._config.nms_iou_threshold,
            imgsz=self._config.image_size,
            device=self._config.device,
            verbose=False,
        )
        detections = self._convert_result(results[0]) if results else []
        return ObjectDetectionResult(
            image=ImageDimensions(width=image.width, height=image.height),
            detections=detections,
            latency_ms=round((perf_counter_ns() - started_at) / 1_000_000),
        )

    def _get_model(self) -> YoloModel:
        """Load and prompt the model once per detector instance."""

        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    model = cast(YoloModel, self._model_factory(self._config.model_source))
                    model.set_classes([target.prompt for target in self._config.targets.values()])
                    self._model = model
        return self._model

    def _convert_result(self, result: PredictionResult) -> list[Detection]:
        boxes = result.boxes
        if boxes is None:
            return []

        coordinates = boxes.xyxyn.detach().cpu().tolist()
        class_ids = boxes.cls.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()

        detections: list[Detection] = []
        for raw_box, class_id, confidence in zip(coordinates, class_ids, confidences, strict=True):
            label = result.names[int(class_id)]
            object_type = self._object_type_by_prompt.get(label.casefold())
            if object_type is None or confidence < self._config.targets[object_type].confidence_threshold:
                continue

            bounding_box = normalized_bounding_box(raw_box)
            if bounding_box is None:
                continue
            detections.append(
                Detection(
                    object_type=object_type,
                    confidence=confidence,
                    bounding_box=bounding_box,
                )
            )
        return sorted(detections, key=lambda detection: detection.confidence, reverse=True)


def _create_yolo_model(source: str) -> object:
    """Create the production YOLO model while keeping the dependency lazy."""

    from ultralytics import YOLOWorld  # pyright: ignore[reportPrivateImportUsage] - documented public API

    return YOLOWorld(source)
