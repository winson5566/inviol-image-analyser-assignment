"""Safety rule for people missing required personal protective equipment."""

from typing import final

from inviol_image_analyser_assignment.config import MissingPpeRuleConfig
from inviol_image_analyser_assignment.models import (
    BoundingBox,
    Detection,
    ObjectDetectionResult,
    ObjectType,
    SafetyEvent,
)
from inviol_image_analyser_assignment.services.object_detection.geometry import bounding_box_overlap_fraction


@final
class MissingPpeRule:
    """Flag each detected person who cannot be associated with all required PPE."""

    rule_type: str = "missing_required_ppe"

    def __init__(self, config: MissingPpeRuleConfig) -> None:
        self._config = config

    def evaluate(self, detection_result: ObjectDetectionResult) -> list[SafetyEvent]:
        """Return one event for each person missing any required PPE."""

        people = [detection for detection in detection_result.detections if detection.object_type is ObjectType.PERSON]
        forklifts = [
            detection for detection in detection_result.detections if detection.object_type is ObjectType.FORKLIFT
        ]
        associated_ppe: list[list[Detection]] = [[] for _ in people]

        for ppe in detection_result.detections:
            if ppe.object_type not in self._config.required_ppe:
                continue
            best_match = self._best_matching_person(ppe, people)
            if best_match is not None:
                associated_ppe[best_match].append(ppe)

        events: list[SafetyEvent] = []
        for person, matched_ppe in zip(people, associated_ppe, strict=True):
            if self._is_forklift_operator(person, forklifts):
                continue
            matched_types = {ppe.object_type for ppe in matched_ppe}
            detected_types = [object_type for object_type in self._config.required_ppe if object_type in matched_types]
            missing_types = [
                object_type for object_type in self._config.required_ppe if object_type not in matched_types
            ]
            if not missing_types:
                continue
            events.append(
                SafetyEvent(
                    rule_type=self.rule_type,
                    subject=person,
                    related_objects=matched_ppe,
                    attributes={
                        "required_ppe": [object_type.value for object_type in self._config.required_ppe],
                        "detected_ppe": [object_type.value for object_type in detected_types],
                        "missing_ppe": [object_type.value for object_type in missing_types],
                    },
                )
            )
        return events

    def _is_forklift_operator(self, person: Detection, forklifts: list[Detection]) -> bool:
        if not self._config.exclude_forklift_operators:
            return False
        return any(
            bounding_box_overlap_fraction(person.bounding_box, forklift.bounding_box)
            >= self._config.minimum_person_overlap_with_forklift
            for forklift in forklifts
        )

    def _best_matching_person(self, ppe: Detection, people: list[Detection]) -> int | None:
        overlaps = [
            (_overlap_with_person(ppe.bounding_box, person.bounding_box), -index, index)
            for index, person in enumerate(people)
        ]
        if not overlaps:
            return None
        overlap, _, person_index = max(overlaps)
        if overlap < self._config.minimum_ppe_overlap_with_person:
            return None
        return person_index


def _overlap_with_person(ppe: BoundingBox, person: BoundingBox) -> float:
    """Return the fraction of a PPE box contained by a person's box."""

    return bounding_box_overlap_fraction(ppe, person)
