"""Tests for missing-PPE safety-rule evaluation."""

from inviol_image_analyser_assignment.config import MissingPpeRuleConfig
from inviol_image_analyser_assignment.models import (
    BoundingBox,
    Detection,
    ImageDimensions,
    ObjectDetectionResult,
    ObjectType,
)
from inviol_image_analyser_assignment.services.safety_detection import MissingPpeRule, SafetyRuleEngine


def detection(object_type: ObjectType, box: tuple[float, float, float, float]) -> Detection:
    return Detection(
        object_type=object_type,
        confidence=0.9,
        bounding_box=BoundingBox(x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]),
    )


def detection_result(*detections: Detection) -> ObjectDetectionResult:
    return ObjectDetectionResult(
        image=ImageDimensions(width=1000, height=800),
        detections=list(detections),
        latency_ms=10,
    )


def missing_ppe_rule() -> MissingPpeRule:
    return MissingPpeRule(
        MissingPpeRuleConfig(
            required_ppe=(ObjectType.SAFETY_HAT, ObjectType.SAFETY_VEST),
            minimum_ppe_overlap_with_person=0.5,
        )
    )


def test_returns_no_event_when_person_has_required_ppe() -> None:
    person = detection(ObjectType.PERSON, (0.1, 0.1, 0.5, 0.9))
    hat = detection(ObjectType.SAFETY_HAT, (0.2, 0.12, 0.3, 0.25))
    vest = detection(ObjectType.SAFETY_VEST, (0.18, 0.35, 0.42, 0.7))

    events = missing_ppe_rule().evaluate(detection_result(person, hat, vest))

    assert events == []


def test_reports_person_missing_required_ppe() -> None:
    person = detection(ObjectType.PERSON, (0.1, 0.1, 0.5, 0.9))
    hat = detection(ObjectType.SAFETY_HAT, (0.2, 0.12, 0.3, 0.25))

    events = missing_ppe_rule().evaluate(detection_result(person, hat))

    assert len(events) == 1
    assert events[0].rule_type == "missing_required_ppe"
    assert events[0].subject == person
    assert events[0].related_objects == [hat]
    assert events[0].attributes == {
        "required_ppe": ["safety_hat", "safety_vest"],
        "detected_ppe": ["safety_hat"],
        "missing_ppe": ["safety_vest"],
    }


def test_engine_returns_image_and_rule_events() -> None:
    person = detection(ObjectType.PERSON, (0.1, 0.1, 0.5, 0.9))
    result = detection_result(person)

    evaluation = SafetyRuleEngine([missing_ppe_rule()]).evaluate(result)

    assert evaluation.image == result.image
    assert len(evaluation.events) == 1
    assert evaluation.events[0].subject == person
    assert evaluation.events[0].attributes["missing_ppe"] == ["safety_hat", "safety_vest"]
