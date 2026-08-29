"""Structured response models for completed image risk assessments."""

from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_serializer

from inviol_image_analyser_assignment.models.object_detection import Detection, ImageDimensions


class RiskLevel(StrEnum):
    """Risk categories derived from numeric scores."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RiskRating(BaseModel):
    """A numeric risk score and its corresponding category."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    score: float = Field(ge=0.0, le=10.0)
    level: RiskLevel

    @field_serializer("score", when_used="json")
    def serialize_score(self, value: float) -> float:
        """Limit API risk scores to two decimal places."""

        return round(value, 2)


class RiskEvent(BaseModel):
    """Risk rating and detection evidence for one safety event."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    rule_type: str = Field(min_length=1)
    risk: RiskRating
    subject: Detection
    related_objects: list[Detection] = Field(default_factory=list)
    attributes: dict[str, JsonValue] = Field(default_factory=dict)


class AnalysisResult(BaseModel):
    """Complete structured risk assessment for one image."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    image: ImageDimensions
    overall_risk: RiskRating
    detected_objects: list[Detection] = Field(default_factory=list)
    events: list[RiskEvent] = Field(default_factory=list)
