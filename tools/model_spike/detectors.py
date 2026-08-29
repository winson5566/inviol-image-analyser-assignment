# pyright: basic
"""Adapters for the candidate object-detection models."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

from PIL import Image

from tools.model_spike.config import (
    CLIP_WEIGHTS_DIRECTORY,
    MODEL_SPEC_BY_NAME,
    MODELS_DIRECTORY,
)


@contextmanager
def _clip_weights_in_project_cache() -> Iterator[None]:
    """Make Ultralytics reuse or download CLIP in the project cache."""

    import clip

    clip_module: Any = clip
    original_load = clip_module.load

    def load_from_project_cache(
        name: str,
        device: Any = None,
        jit: bool = False,
        download_root: str | None = None,
    ) -> Any:
        # Ignore Ultralytics' requested cache so the spike and application share
        # one predictable CLIP weights directory.
        del download_root
        return original_load(
            name,
            device=device,
            jit=jit,
            download_root=str(CLIP_WEIGHTS_DIRECTORY),
        )

    clip_module.load = load_from_project_cache
    try:
        yield
    finally:
        clip_module.load = original_load


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """A pixel-space bounding box in left, top, right, bottom order."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def clamped(self, width: int, height: int) -> BoundingBox:
        """Return a copy constrained to an image's valid pixel coordinates."""

        return BoundingBox(
            x_min=max(0.0, min(self.x_min, float(width - 1))),
            y_min=max(0.0, min(self.y_min, float(height - 1))),
            x_max=max(0.0, min(self.x_max, float(width - 1))),
            y_max=max(0.0, min(self.y_max, float(height - 1))),
        )

    @property
    def area(self) -> float:
        """Return the non-negative area of the box."""

        return max(0.0, self.x_max - self.x_min) * max(0.0, self.y_max - self.y_min)

    def intersection_over_union(self, other: BoundingBox) -> float:
        """Calculate intersection-over-union with another pixel-space box."""

        intersection_width = max(0.0, min(self.x_max, other.x_max) - max(self.x_min, other.x_min))
        intersection_height = max(0.0, min(self.y_max, other.y_max) - max(self.y_min, other.y_min))
        intersection = intersection_width * intersection_height
        union = self.area + other.area - intersection
        return intersection / union if union > 0.0 else 0.0


@dataclass(frozen=True, slots=True)
class Detection:
    """A normalized detection produced by any candidate model library."""

    label: str
    confidence: float
    box: BoundingBox

    def to_dict(self) -> dict[str, Any]:
        """Convert the detection into a JSON-serializable dictionary."""

        return {
            "label": self.label,
            "confidence": round(self.confidence, 6),
            "box": asdict(self.box),
        }


def class_aware_nms(detections: Sequence[Detection], iou_threshold: float) -> list[Detection]:
    """Suppress lower-confidence overlapping boxes for the same class label."""

    kept: list[Detection] = []
    for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
        overlaps_existing = any(
            candidate.label.casefold() == existing.label.casefold()
            and candidate.box.intersection_over_union(existing.box) > iou_threshold
            for existing in kept
        )
        if not overlaps_existing:
            kept.append(candidate)
    return kept


class Detector(ABC):
    """Library-independent interface used by the batch evaluation runner."""

    name: str
    display_name: str
    threshold: float

    @abstractmethod
    def load(self) -> None:
        """Load model weights and any prompt encoders."""

    @abstractmethod
    def detect(self, image: Image.Image) -> list[Detection]:
        """Run inference and return detections in a shared representation."""

    def close(self) -> None:
        """Release model memory before the next candidate is loaded."""

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass


class UltralyticsDetector(Detector):
    """Run either fixed-vocabulary YOLOv8n or prompted YOLO-World."""

    def __init__(
        self,
        *,
        name: str,
        display_name: str,
        weights: str,
        threshold: float,
        device: str,
        prompts: Sequence[str] | None = None,
        class_ids: Sequence[int] | None = None,
    ) -> None:
        self.name = name
        self.display_name = display_name
        self.weights = weights
        self.threshold = threshold
        self.device = device
        self.prompts = list(prompts) if prompts is not None else None
        self.class_ids = list(class_ids) if class_ids is not None else None
        self._model: Any | None = None

    def load(self) -> None:
        """Load Ultralytics weights and configure an optional vocabulary."""

        from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage] - documented public API

        model = YOLO(self.weights)
        if self.prompts is not None:
            set_classes = model.set_classes
            if set_classes is None:
                raise RuntimeError(f"{self.display_name} does not support custom class prompts")
            with _clip_weights_in_project_cache():
                set_classes(self.prompts)
        self._model = model

    def detect(self, image: Image.Image) -> list[Detection]:
        """Convert an Ultralytics result into the common detection model."""

        if self._model is None:
            raise RuntimeError(f"{self.display_name} is not loaded")

        results = self._model.predict(
            source=image,
            conf=self.threshold,
            device=self.device,
            classes=self.class_ids,
            verbose=False,
        )
        result = results[0]
        if result.boxes is None:
            return []

        coordinates = result.boxes.xyxy.detach().cpu().tolist()
        class_ids = result.boxes.cls.detach().cpu().tolist()
        confidences = result.boxes.conf.detach().cpu().tolist()

        detections: list[Detection] = []
        for raw_box, class_id, confidence in zip(coordinates, class_ids, confidences, strict=True):
            label = str(result.names[int(class_id)])
            detections.append(
                Detection(
                    label=label,
                    confidence=float(confidence),
                    box=BoundingBox(
                        x_min=float(raw_box[0]),
                        y_min=float(raw_box[1]),
                        x_max=float(raw_box[2]),
                        y_max=float(raw_box[3]),
                    ),
                )
            )
        return detections

    def close(self) -> None:
        """Drop references to Ultralytics/PyTorch model state."""

        self._model = None
        super().close()


class TransformersZeroShotDetector(Detector):
    """Run a Transformers zero-shot object-detection pipeline."""

    def __init__(
        self,
        *,
        name: str,
        display_name: str,
        model_id: str,
        threshold: float,
        device: str,
        prompts: Sequence[str],
    ) -> None:
        self.name = name
        self.display_name = display_name
        self.model_id = model_id
        self.threshold = threshold
        self.device = device
        self.prompts = list(prompts)
        self._pipeline: Any | None = None

    def load(self) -> None:
        """Build a Hugging Face zero-shot object-detection pipeline."""

        from transformers import pipeline

        self._pipeline = pipeline(
            task="zero-shot-object-detection",
            model=self.model_id,
            device=self.device,
        )

    def detect(self, image: Image.Image) -> list[Detection]:
        """Convert a Transformers pipeline result into common detections."""

        if self._pipeline is None:
            raise RuntimeError(f"{self.display_name} is not loaded")

        predictions = self._pipeline(
            image,
            candidate_labels=self.prompts,
            threshold=self.threshold,
        )

        detections: list[Detection] = []
        for prediction in predictions:
            raw_box = prediction["box"]
            detections.append(
                Detection(
                    label=str(prediction["label"]),
                    confidence=float(prediction["score"]),
                    box=BoundingBox(
                        x_min=float(raw_box["xmin"]),
                        y_min=float(raw_box["ymin"]),
                        x_max=float(raw_box["xmax"]),
                        y_max=float(raw_box["ymax"]),
                    ),
                )
            )
        return detections

    def close(self) -> None:
        """Drop references to Transformers/PyTorch model state."""

        self._pipeline = None
        super().close()


def build_detector(
    name: str,
    *,
    device: str,
    prompts: Sequence[str],
    threshold_override: float | None = None,
) -> Detector:
    """Construct one of the supported model candidates by its CLI name."""

    try:
        spec = MODEL_SPEC_BY_NAME[name]
    except KeyError as error:
        raise ValueError(f"Unsupported detector: {name}") from error

    threshold = threshold_override if threshold_override is not None else spec.threshold
    if spec.backend == "ultralytics":
        return UltralyticsDetector(
            name=name,
            display_name=spec.display_name,
            weights=str(MODELS_DIRECTORY / spec.source),
            threshold=threshold,
            device=device,
            prompts=prompts if spec.uses_prompts else None,
            class_ids=spec.class_ids,
        )
    return TransformersZeroShotDetector(
        name=name,
        display_name=spec.display_name,
        model_id=spec.source,
        threshold=threshold,
        device=device,
        prompts=prompts,
    )
