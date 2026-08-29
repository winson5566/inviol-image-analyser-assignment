"""Typed configuration for image analysis services."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


class MissingPpeRuleConfig(BaseModel):
    """Settings used to associate required PPE with detected people."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    required_ppe: tuple[ObjectType, ...] = Field(min_length=1)
    minimum_ppe_overlap_with_person: float = Field(ge=0.0, le=1.0)

    @field_validator("required_ppe")
    @classmethod
    def validate_required_ppe(cls, required_ppe: tuple[ObjectType, ...]) -> tuple[ObjectType, ...]:
        """Accept only unique PPE object types supported by the detector."""

        allowed_ppe = {ObjectType.SAFETY_HAT, ObjectType.SAFETY_VEST}
        invalid_types = [object_type for object_type in required_ppe if object_type not in allowed_ppe]
        if invalid_types:
            invalid_values = ", ".join(object_type.value for object_type in invalid_types)
            raise ValueError(f"required PPE contains unsupported object types: {invalid_values}")
        if len(required_ppe) != len(set(required_ppe)):
            raise ValueError("required PPE object types must be unique")
        return required_ppe


class PersonForkliftProximityRuleConfig(BaseModel):
    """Settings used to identify unsafe person-forklift proximity."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    minimum_person_overlap_with_forklift: float = Field(gt=0.0, le=1.0)
    maximum_normalized_distance: float = Field(gt=0.0, le=1.0)


class SafetyRulesConfig(BaseModel):
    """Runtime settings for the workplace safety-rule engine."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    missing_ppe: MissingPpeRuleConfig
    person_forklift_proximity: PersonForkliftProximityRuleConfig


class AnalysisConfig(BaseModel):
    """Top-level configuration for the image-analysis pipeline."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    object_detection: ObjectDetectionConfig
    safety_rules: SafetyRulesConfig


class AnalysisConfigReference(BaseModel):
    """Pointer from analysis settings to an immutable object-detection release."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    object_detection_config: Path
    safety_rules_config: Path


def load_object_detection_config(path: Path) -> ObjectDetectionConfig:
    """Load and validate one versioned detector release configuration."""

    return ObjectDetectionConfig.model_validate_json(path.read_text(encoding="utf-8"))


def load_safety_rules_config(path: Path) -> SafetyRulesConfig:
    """Load and validate one versioned safety-rules configuration."""

    return SafetyRulesConfig.model_validate_json(path.read_text(encoding="utf-8"))


def load_analysis_config(path: Path) -> AnalysisConfig:
    """Load the active detector and safety-rule releases selected for analysis."""

    reference = AnalysisConfigReference.model_validate_json(path.read_text(encoding="utf-8"))
    object_detection_path = _resolve_config_path(path, reference.object_detection_config)
    safety_rules_path = _resolve_config_path(path, reference.safety_rules_config)
    return AnalysisConfig(
        object_detection=load_object_detection_config(object_detection_path),
        safety_rules=load_safety_rules_config(safety_rules_path),
    )


def _resolve_config_path(reference_path: Path, selected_path: Path) -> Path:
    """Resolve a selected release relative to its top-level configuration."""

    return selected_path if selected_path.is_absolute() else reference_path.parent / selected_path
