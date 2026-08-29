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
    exclude_forklift_operators: bool
    minimum_person_overlap_with_forklift: float = Field(gt=0.0, le=1.0)

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


class RiskLevelThresholdsConfig(BaseModel):
    """Minimum scores required to enter the medium and high risk levels."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    medium_min_score: float = Field(gt=0.0, le=10.0)
    high_min_score: float = Field(gt=0.0, le=10.0)

    @model_validator(mode="after")
    def validate_threshold_order(self) -> RiskLevelThresholdsConfig:
        """Require the medium threshold to be lower than the high threshold."""

        if self.medium_min_score >= self.high_min_score:
            raise ValueError("medium risk minimum must be less than high risk minimum")
        return self


class MissingPpeRiskConfig(BaseModel):
    """Risk scores assigned to missing required PPE."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    missing_safety_hat_score: float = Field(ge=0.0, le=10.0)
    missing_safety_vest_score: float = Field(ge=0.0, le=10.0)
    multiple_missing_ppe_score: float = Field(ge=0.0, le=10.0)

    @model_validator(mode="after")
    def validate_multiple_missing_ppe_score(self) -> MissingPpeRiskConfig:
        """Prevent multiple missing items from reducing the assessed risk."""

        if self.multiple_missing_ppe_score < max(
            self.missing_safety_hat_score,
            self.missing_safety_vest_score,
        ):
            raise ValueError("multiple-missing PPE score must not be lower than an individual PPE score")
        return self


class PersonForkliftProximityRiskConfig(BaseModel):
    """Risk score assigned to unsafe person-forklift proximity."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    score: float = Field(ge=0.0, le=10.0)


class RiskAssessmentConfig(BaseModel):
    """Runtime policy used to assess safety events."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    risk_level_thresholds: RiskLevelThresholdsConfig
    missing_ppe: MissingPpeRiskConfig
    person_forklift_proximity: PersonForkliftProximityRiskConfig


class SafetyRulesConfig(BaseModel):
    """Runtime settings for the workplace safety-rule engine."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    missing_ppe: MissingPpeRuleConfig
    person_forklift_proximity: PersonForkliftProximityRuleConfig


class MissingPpeRuleReleaseConfig(MissingPpeRuleConfig):
    """Versioned missing-PPE rule settings and risk policy."""

    risk_assessment: MissingPpeRiskConfig


class PersonForkliftProximityRuleReleaseConfig(PersonForkliftProximityRuleConfig):
    """Versioned proximity-rule settings and risk policy."""

    risk_assessment: PersonForkliftProximityRiskConfig


class SafetyRulesReleaseConfig(BaseModel):
    """Complete versioned safety-rule and risk-assessment configuration."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    risk_level_thresholds: RiskLevelThresholdsConfig
    missing_ppe: MissingPpeRuleReleaseConfig
    person_forklift_proximity: PersonForkliftProximityRuleReleaseConfig


class AnalysisConfig(BaseModel):
    """Top-level configuration for the image-analysis pipeline."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    object_detection: ObjectDetectionConfig
    safety_rules: SafetyRulesConfig
    risk_assessment: RiskAssessmentConfig


class AnalysisConfigReference(BaseModel):
    """Pointers from analysis settings to versioned service configurations."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    object_detection_config: Path
    safety_rules_config: Path


def load_object_detection_config(path: Path) -> ObjectDetectionConfig:
    """Load and validate one versioned detector release configuration."""

    return ObjectDetectionConfig.model_validate_json(path.read_text(encoding="utf-8"))


def load_safety_rules_config(path: Path) -> SafetyRulesReleaseConfig:
    """Load and validate one versioned safety-rules configuration."""

    return SafetyRulesReleaseConfig.model_validate_json(path.read_text(encoding="utf-8"))


def load_analysis_config(path: Path) -> AnalysisConfig:
    """Load the active detector, safety-rule, and risk-assessment configuration."""

    reference = AnalysisConfigReference.model_validate_json(path.read_text(encoding="utf-8"))
    object_detection_path = _resolve_config_path(path, reference.object_detection_config)
    safety_rules_path = _resolve_config_path(path, reference.safety_rules_config)
    safety_rules_release = load_safety_rules_config(safety_rules_path)
    return AnalysisConfig(
        object_detection=load_object_detection_config(object_detection_path),
        safety_rules=SafetyRulesConfig(
            missing_ppe=safety_rules_release.missing_ppe,
            person_forklift_proximity=safety_rules_release.person_forklift_proximity,
        ),
        risk_assessment=RiskAssessmentConfig(
            risk_level_thresholds=safety_rules_release.risk_level_thresholds,
            missing_ppe=safety_rules_release.missing_ppe.risk_assessment,
            person_forklift_proximity=safety_rules_release.person_forklift_proximity.risk_assessment,
        ),
    )


def _resolve_config_path(reference_path: Path, selected_path: Path) -> Path:
    """Resolve a selected release relative to its top-level configuration."""

    return selected_path if selected_path.is_absolute() else reference_path.parent / selected_path
