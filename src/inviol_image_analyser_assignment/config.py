"""Typed configuration for image analysis services."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from inviol_image_analyser_assignment.models.object_detection import ObjectType


class DetectorType(StrEnum):
    """Supported object-detection adapter implementations."""

    YOLO_WORLD = "yolo_world"
    GROUNDING_DINO = "grounding_dino"


class DetectionTargetConfig(BaseModel):
    """Prompt and minimum accepted confidence for one object type."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    prompt: str = Field(min_length=1)
    confidence_threshold: float = Field(ge=0.0, le=1.0)


class ObjectDetectionConfig(BaseModel):
    """Runtime settings for the object-detection adapter."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    detector_type: DetectorType
    model_source: str = Field(min_length=1)
    device: str = Field(default="cpu", min_length=1)
    image_size: int = Field(default=640, ge=32)
    nms_iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    targets: dict[ObjectType, DetectionTargetConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_prompts(self) -> ObjectDetectionConfig:
        """Ensure model labels map unambiguously back to domain object types."""

        prompts = [target.prompt.casefold() for target in self.targets.values()]
        if len(prompts) != len(set(prompts)):
            raise ValueError("object-detection prompts must be unique")
        return self


class AnalysisConfig(BaseModel):
    """Top-level configuration for the image-analysis pipeline."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    object_detection: ObjectDetectionConfig


class AnalysisConfigReference(BaseModel):
    """Pointer from analysis settings to an immutable object-detection release."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    object_detection_config: Path


def load_object_detection_config(path: Path) -> ObjectDetectionConfig:
    """Load and validate one versioned detector release configuration."""

    return ObjectDetectionConfig.model_validate_json(path.read_text(encoding="utf-8"))


def load_analysis_config(path: Path) -> AnalysisConfig:
    """Load the active detector release selected by the analysis configuration."""

    reference = AnalysisConfigReference.model_validate_json(path.read_text(encoding="utf-8"))
    object_detection_path = reference.object_detection_config
    if not object_detection_path.is_absolute():
        object_detection_path = path.parent / object_detection_path
    return AnalysisConfig(object_detection=load_object_detection_config(object_detection_path))
