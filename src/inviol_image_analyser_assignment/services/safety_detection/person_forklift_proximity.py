"""Safety rule for people standing too close to forklifts."""

from math import hypot
from typing import final

from inviol_image_analyser_assignment.config import PersonForkliftProximityRuleConfig
from inviol_image_analyser_assignment.models import (
    BoundingBox,
    ImageDimensions,
    ObjectDetectionResult,
    ObjectType,
    SafetyEvent,
)


@final
class PersonForkliftProximityRule:
    """Flag person-forklift pairs whose bounding boxes are too close."""

    rule_type: str = "person_near_forklift"

    def __init__(self, config: PersonForkliftProximityRuleConfig) -> None:
        self._config = config

    def evaluate(self, detection_result: ObjectDetectionResult) -> list[SafetyEvent]:
        """Return one event for each person-forklift pair within the configured distance."""

        people = [detection for detection in detection_result.detections if detection.object_type is ObjectType.PERSON]
        forklifts = [
            detection for detection in detection_result.detections if detection.object_type is ObjectType.FORKLIFT
        ]

        events: list[SafetyEvent] = []
        for person in people:
            for forklift in forklifts:
                if (
                    _person_overlap_with_forklift(person.bounding_box, forklift.bounding_box)
                    >= self._config.minimum_person_overlap_with_forklift
                ):
                    continue
                distance = _normalized_bounding_box_distance(
                    person.bounding_box,
                    forklift.bounding_box,
                    detection_result.image,
                )
                if distance > self._config.maximum_normalized_distance:
                    continue
                events.append(
                    SafetyEvent(
                        rule_type=self.rule_type,
                        subject=person,
                        related_objects=[forklift],
                        attributes={
                            "normalized_distance": round(distance, 4),
                            "maximum_normalized_distance": self._config.maximum_normalized_distance,
                        },
                    )
                )
        return events


def _person_overlap_with_forklift(person: BoundingBox, forklift: BoundingBox) -> float:
    """Return the fraction of a person's box contained by a forklift's box."""

    intersection_width = max(0.0, min(person.x_max, forklift.x_max) - max(person.x_min, forklift.x_min))
    intersection_height = max(0.0, min(person.y_max, forklift.y_max) - max(person.y_min, forklift.y_min))
    intersection = intersection_width * intersection_height
    person_area = (person.x_max - person.x_min) * (person.y_max - person.y_min)
    return intersection / person_area


def _normalized_bounding_box_distance(
    first: BoundingBox,
    second: BoundingBox,
    image: ImageDimensions,
) -> float:
    """Measure the shortest box-edge distance as a fraction of the image diagonal."""

    horizontal_gap = max(first.x_min - second.x_max, second.x_min - first.x_max, 0.0)
    vertical_gap = max(first.y_min - second.y_max, second.y_min - first.y_max, 0.0)
    horizontal_distance = horizontal_gap * image.width
    vertical_distance = vertical_gap * image.height
    return hypot(horizontal_distance, vertical_distance) / hypot(image.width, image.height)
