"""Domain models produced by workplace safety detection."""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from inviol_image_analyser_assignment.models.object_detection import Detection, ImageDimensions


class SafetyEvent(BaseModel):
    """A safety condition and the detections that caused it."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    rule_type: str = Field(min_length=1)
    subject: Detection
    related_objects: list[Detection] = Field(default_factory=list)
    attributes: dict[str, JsonValue] = Field(default_factory=dict)


class SafetyDetectionResult(BaseModel):
    """Safety events detected in one image."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    image: ImageDimensions
    events: list[SafetyEvent] = Field(default_factory=list)
