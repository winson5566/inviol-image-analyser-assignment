"""Central configuration for the object-detection model spike."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

MODEL_SPIKE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODEL_SPIKE_DIRECTORY.parents[1]
MODELS_DIRECTORY = MODEL_SPIKE_DIRECTORY / "models"
CLIP_WEIGHTS_DIRECTORY = REPOSITORY_ROOT / "weights" / "clip"
REPORTS_DIRECTORY = MODEL_SPIKE_DIRECTORY / "reports"
SAMPLE_IMAGES_DIRECTORY = REPOSITORY_ROOT / "sample_images"

DEFAULT_PROMPTS = ("person", "material handling vehicle", "safety hat", "safety vest")
SUPPORTED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Static configuration needed to construct one candidate detector."""

    name: str
    display_name: str
    backend: Literal["ultralytics", "transformers"]
    source: str
    threshold: float
    class_ids: tuple[int, ...] | None = None
    uses_prompts: bool = True


MODEL_SPECS = (
    ModelSpec(
        name="yolov8n",
        display_name="YOLOv8n (COCO baseline)",
        backend="ultralytics",
        source="yolov8n.pt",
        threshold=0.03,
        class_ids=(0,),
        uses_prompts=False,
    ),
    ModelSpec(
        name="yolo_world",
        display_name="YOLOv8s-World-v2",
        backend="ultralytics",
        source="yolov8s-worldv2.pt",
        threshold=0.035,
    ),
    ModelSpec(
        name="owlv2",
        display_name="OWLv2 base patch16 ensemble",
        backend="transformers",
        source="google/owlv2-base-patch16-ensemble",
        threshold=0.10,
    ),
    ModelSpec(
        name="grounding_dino",
        display_name="Grounding DINO tiny",
        backend="transformers",
        source="IDEA-Research/grounding-dino-tiny",
        threshold=0.10,
    ),
)

MODEL_SPEC_BY_NAME = {spec.name: spec for spec in MODEL_SPECS}
DEFAULT_MODELS = tuple(MODEL_SPEC_BY_NAME)
