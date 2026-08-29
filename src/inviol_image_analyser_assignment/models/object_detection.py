"""Domain models produced by object detection."""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator


class ObjectType(StrEnum):
    """Workplace object categories supported by the safety rules."""

    PERSON = "person"
    FORKLIFT = "forklift"
    SAFETY_HAT = "safety_hat"
    SAFETY_VEST = "safety_vest"


class BoundingBox(BaseModel):
    """Image-size-independent bounding box using normalized coordinates."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    x_min: float = Field(ge=0.0, le=1.0)
    y_min: float = Field(ge=0.0, le=1.0)
    x_max: float = Field(ge=0.0, le=1.0)
    y_max: float = Field(ge=0.0, le=1.0)

    @field_serializer("x_min", "y_min", "x_max", "y_max", when_used="json")
    def serialize_coordinate(self, value: float) -> float:
        """Limit API coordinates to four decimal places."""

        return round(value, 4)

    @model_validator(mode="after")
    def validate_dimensions(self) -> BoundingBox:
        """Reject empty or inverted boxes from malformed model output."""

        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("bounding box maximums must be greater than minimums")
        return self


class Detection(BaseModel):
    """A normalized object detection suitable for rules and API responses."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    object_type: ObjectType
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: BoundingBox

    @field_serializer("confidence", when_used="json")
    def serialize_confidence(self, value: float) -> float:
        """Limit API confidence values to three decimal places."""

        return round(value, 3)


class ImageDimensions(BaseModel):
    """Dimensions of the image used for object detection."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    width: int = Field(gt=0)
    height: int = Field(gt=0)


class ObjectDetectionResult(BaseModel):
    """Self-contained object-detection result for one image."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    image: ImageDimensions
    detections: list[Detection] = Field(default_factory=list)
    latency_ms: int = Field(ge=0)
