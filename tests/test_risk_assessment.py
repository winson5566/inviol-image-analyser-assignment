"""Tests for converting safety events into structured risk assessments."""

from inviol_image_analyser_assignment.config import (
    MissingPpeRiskConfig,
    PersonForkliftProximityRiskConfig,
    RiskAssessmentConfig,
    RiskLevelThresholdsConfig,
)
from inviol_image_analyser_assignment.models import (
    BoundingBox,
    Detection,
    ImageDimensions,
    ObjectDetectionResult,
    ObjectType,
    RiskLevel,
    RiskRating,
    SafetyDetectionResult,
    SafetyEvent,
)
from inviol_image_analyser_assignment.services.risk_assessment import RiskAssessmentService

_IMAGE = ImageDimensions(width=1000, height=800)
_RISK_CONFIG = RiskAssessmentConfig(
    risk_level_thresholds=RiskLevelThresholdsConfig(medium_min_score=4, high_min_score=7),
    missing_ppe=MissingPpeRiskConfig(
        missing_safety_hat_score=8,
        missing_safety_vest_score=6,
        multiple_missing_ppe_score=9,
    ),
    person_forklift_proximity=PersonForkliftProximityRiskConfig(score=9),
)


def _detection(object_type: ObjectType, box: tuple[float, float, float, float]) -> Detection:
    return Detection(
        object_type=object_type,
        confidence=0.9,
        bounding_box=BoundingBox(x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]),
    )


def test_assesses_safety_events_and_uses_highest_overall_risk() -> None:
    person = _detection(ObjectType.PERSON, (0.1, 0.1, 0.4, 0.9))
    forklift = _detection(ObjectType.FORKLIFT, (0.45, 0.2, 0.8, 0.9))
    detections = ObjectDetectionResult(image=_IMAGE, detections=[person, forklift], latency_ms=10)
    safety = SafetyDetectionResult(
        image=_IMAGE,
        events=[
            SafetyEvent(
                rule_type="missing_required_ppe",
                subject=person,
                attributes={"missing_ppe": ["safety_vest"]},
            ),
            SafetyEvent(
                rule_type="person_near_forklift",
                subject=person,
                related_objects=[forklift],
                attributes={"normalized_distance": 0.03},
            ),
        ],
    )

    result = RiskAssessmentService(_RISK_CONFIG).assess(detections, safety)

    assert result.image == _IMAGE
    assert result.detected_objects == [person, forklift]
    assert result.overall_risk == RiskRating(score=9, level=RiskLevel.HIGH)
    assert [event.risk for event in result.events] == [
        RiskRating(score=6, level=RiskLevel.MEDIUM),
        RiskRating(score=9, level=RiskLevel.HIGH),
    ]
    assert result.events[0].rule_type == "missing_required_ppe"
    assert result.events[1].rule_type == "person_near_forklift"
    assert result.events[1].related_objects == [forklift]


def test_uses_combined_score_when_multiple_ppe_items_are_missing() -> None:
    person = _detection(ObjectType.PERSON, (0.1, 0.1, 0.4, 0.9))
    detections = ObjectDetectionResult(image=_IMAGE, detections=[person], latency_ms=10)
    safety = SafetyDetectionResult(
        image=_IMAGE,
        events=[
            SafetyEvent(
                rule_type="missing_required_ppe",
                subject=person,
                attributes={"missing_ppe": ["safety_hat", "safety_vest"]},
            )
        ],
    )

    result = RiskAssessmentService(_RISK_CONFIG).assess(detections, safety)

    assert result.overall_risk == RiskRating(score=9, level=RiskLevel.HIGH)
    assert result.events[0].attributes["missing_ppe"] == ["safety_hat", "safety_vest"]


def test_returns_low_zero_risk_when_no_safety_events_exist() -> None:
    detections = ObjectDetectionResult(image=_IMAGE, detections=[], latency_ms=10)
    safety = SafetyDetectionResult(image=_IMAGE, events=[])

    result = RiskAssessmentService(_RISK_CONFIG).assess(detections, safety)

    assert result.overall_risk == RiskRating(score=0, level=RiskLevel.LOW)
    assert result.events == []
