"""Tests for workplace safety-rule evaluation."""

from inviol_image_analyser_assignment.config import (
    MissingPpeRuleConfig,
    PersonForkliftProximityRuleConfig,
    SafetyRulesConfig,
)
from inviol_image_analyser_assignment.models import (
    BoundingBox,
    Detection,
    ImageDimensions,
    ObjectDetectionResult,
    ObjectType,
)
from inviol_image_analyser_assignment.services.safety_detection import (
    MissingPpeRule,
    PersonForkliftProximityRule,
    create_safety_rule_engine,
)

_MISSING_PPE_CONFIG = MissingPpeRuleConfig(
    required_ppe=(ObjectType.SAFETY_HAT, ObjectType.SAFETY_VEST),
    minimum_ppe_overlap_with_person=0.5,
)
_PROXIMITY_CONFIG = PersonForkliftProximityRuleConfig(
    minimum_person_overlap_with_forklift=0.8,
    maximum_normalized_distance=0.2,
)
_SAFETY_RULES_CONFIG = SafetyRulesConfig(
    missing_ppe=_MISSING_PPE_CONFIG,
    person_forklift_proximity=_PROXIMITY_CONFIG,
)


def _detection(object_type: ObjectType, box: tuple[float, float, float, float]) -> Detection:
    return Detection(
        object_type=object_type,
        confidence=0.9,
        bounding_box=BoundingBox(x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]),
    )


def _detection_result(*detections: Detection) -> ObjectDetectionResult:
    return ObjectDetectionResult(
        image=ImageDimensions(width=1000, height=800),
        detections=list(detections),
        latency_ms=10,
    )


def test_returns_no_event_when_person_has_required_ppe() -> None:
    person = _detection(ObjectType.PERSON, (0.1, 0.1, 0.5, 0.9))
    hat = _detection(ObjectType.SAFETY_HAT, (0.2, 0.12, 0.3, 0.25))
    vest = _detection(ObjectType.SAFETY_VEST, (0.18, 0.35, 0.42, 0.7))

    events = MissingPpeRule(_MISSING_PPE_CONFIG).evaluate(_detection_result(person, hat, vest))

    assert events == []


def test_reports_person_missing_required_ppe() -> None:
    person = _detection(ObjectType.PERSON, (0.1, 0.1, 0.5, 0.9))
    hat = _detection(ObjectType.SAFETY_HAT, (0.2, 0.12, 0.3, 0.25))

    events = MissingPpeRule(_MISSING_PPE_CONFIG).evaluate(_detection_result(person, hat))

    assert len(events) == 1
    assert events[0].rule_type == "missing_required_ppe"
    assert events[0].subject == person
    assert events[0].related_objects == [hat]
    assert events[0].attributes == {
        "required_ppe": ["safety_hat", "safety_vest"],
        "detected_ppe": ["safety_hat"],
        "missing_ppe": ["safety_vest"],
    }


def test_engine_reports_unsafe_person_forklift_proximity() -> None:
    person = _detection(ObjectType.PERSON, (0.1, 0.1, 0.5, 0.9))
    hat = _detection(ObjectType.SAFETY_HAT, (0.2, 0.12, 0.3, 0.25))
    vest = _detection(ObjectType.SAFETY_VEST, (0.18, 0.35, 0.42, 0.7))
    forklift = _detection(ObjectType.FORKLIFT, (0.52, 0.3, 0.82, 0.85))
    result = _detection_result(person, hat, vest, forklift)

    evaluation = create_safety_rule_engine(_SAFETY_RULES_CONFIG).evaluate(result)

    assert evaluation.image == result.image
    assert len(evaluation.events) == 1
    assert evaluation.events[0].rule_type == "person_near_forklift"
    assert evaluation.events[0].subject == person
    assert evaluation.events[0].related_objects == [forklift]
    assert evaluation.events[0].attributes == {
        "normalized_distance": 0.0156,
        "maximum_normalized_distance": 0.2,
    }


def test_returns_no_event_when_person_is_far_from_forklift() -> None:
    person = _detection(ObjectType.PERSON, (0.05, 0.1, 0.2, 0.8))
    forklift = _detection(ObjectType.FORKLIFT, (0.7, 0.2, 0.95, 0.75))

    events = PersonForkliftProximityRule(_PROXIMITY_CONFIG).evaluate(_detection_result(person, forklift))

    assert events == []


def test_excludes_forklift_operator_from_proximity_events() -> None:
    operator = _detection(ObjectType.PERSON, (0.3, 0.2, 0.5, 0.7))
    forklift = _detection(ObjectType.FORKLIFT, (0.2, 0.1, 0.6, 0.9))

    events = PersonForkliftProximityRule(_PROXIMITY_CONFIG).evaluate(_detection_result(operator, forklift))

    assert events == []
