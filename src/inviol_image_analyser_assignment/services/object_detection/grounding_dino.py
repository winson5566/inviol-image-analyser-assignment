"""Grounding DINO object-detector adapter."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from threading import Lock
from time import perf_counter_ns
from typing import Protocol, TypedDict, cast, final

from PIL import Image

from inviol_image_analyser_assignment.config import ObjectDetectionConfig
from inviol_image_analyser_assignment.models import Detection, ImageDimensions, ObjectDetectionResult, ObjectType
from inviol_image_analyser_assignment.services.object_detection.geometry import (
    class_aware_nms,
    normalized_bounding_box,
)


class GroundingDinoBox(TypedDict):
    """Pixel-space box returned by a Transformers detection pipeline."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float


class GroundingDinoPrediction(TypedDict):
    """Prediction returned by a Transformers zero-shot detection pipeline."""

    score: float
    label: str
    box: GroundingDinoBox


class GroundingDinoPipeline(Protocol):
    """Transformers pipeline operations consumed by the adapter."""

    def __call__(
        self,
        image: Image.Image,
        *,
        candidate_labels: list[str],
        threshold: float,
    ) -> Sequence[GroundingDinoPrediction]: ...


GroundingDinoPipelineFactory = Callable[[str, str], object]


@final
class GroundingDinoDetector:
    """Detect prompted workplace objects with a lazy Transformers pipeline."""

    def __init__(
        self,
        config: ObjectDetectionConfig,
        pipeline_factory: GroundingDinoPipelineFactory | None = None,
    ) -> None:
        self._config: ObjectDetectionConfig = config
        self._pipeline_factory: GroundingDinoPipelineFactory = pipeline_factory or _create_grounding_dino_pipeline
        self._pipeline: GroundingDinoPipeline | None = None
        self._pipeline_lock: Lock = Lock()
        self._object_type_by_prompt: dict[str, ObjectType] = {
            target.prompt.casefold(): object_type for object_type, target in config.targets.items()
        }

    def detect(self, image: Image.Image) -> ObjectDetectionResult:
        """Run grounded detection and normalize pixel-space model results."""

        started_at = perf_counter_ns()
        predictions = self._get_pipeline()(
            image,
            candidate_labels=[target.prompt for target in self._config.targets.values()],
            threshold=min(target.confidence_threshold for target in self._config.targets.values()),
        )
        width, height = image.size
        detections: list[Detection] = []
        for prediction in predictions:
            object_type = self._object_type_by_prompt.get(prediction["label"].casefold())
            confidence = float(prediction["score"])
            if object_type is None or confidence < self._config.targets[object_type].confidence_threshold:
                continue

            raw_box = prediction["box"]
            bounding_box = normalized_bounding_box(
                [
                    raw_box["xmin"] / width,
                    raw_box["ymin"] / height,
                    raw_box["xmax"] / width,
                    raw_box["ymax"] / height,
                ]
            )
            if bounding_box is None:
                continue
            detections.append(
                Detection(
                    object_type=object_type,
                    confidence=confidence,
                    bounding_box=bounding_box,
                )
            )
        return ObjectDetectionResult(
            image=ImageDimensions(width=width, height=height),
            detections=class_aware_nms(detections, self._config.nms_iou_threshold),
            latency_ms=round((perf_counter_ns() - started_at) / 1_000_000),
        )

    def _get_pipeline(self) -> GroundingDinoPipeline:
        """Load the Transformers inference pipeline once per detector instance."""

        if self._pipeline is None:
            with self._pipeline_lock:
                if self._pipeline is None:
                    self._pipeline = cast(
                        GroundingDinoPipeline,
                        self._pipeline_factory(self._config.model_source, self._config.device),
                    )
        return self._pipeline


def _create_grounding_dino_pipeline(source: str, device: str) -> object:
    """Create the production Grounding DINO pipeline lazily."""

    from transformers import pipeline

    return pipeline(task="zero-shot-object-detection", model=source, device=device)
