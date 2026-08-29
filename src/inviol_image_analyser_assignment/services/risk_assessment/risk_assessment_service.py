"""Convert objective safety events into structured risk assessments."""

from typing import final

from inviol_image_analyser_assignment.config import RiskAssessmentConfig
from inviol_image_analyser_assignment.models import (
    AnalysisResult,
    ObjectDetectionResult,
    RiskEvent,
    RiskLevel,
    RiskRating,
    SafetyDetectionResult,
    SafetyEvent,
)

_MISSING_PPE_RULE = "missing_required_ppe"
_PERSON_NEAR_FORKLIFT_RULE = "person_near_forklift"


@final
class RiskAssessmentService:
    """Apply configurable risk policy to safety-rule events."""

    def __init__(self, config: RiskAssessmentConfig) -> None:
        self._config = config

    def assess(
        self,
        detection_result: ObjectDetectionResult,
        safety_result: SafetyDetectionResult,
    ) -> AnalysisResult:
        """Return risk-rated safety events and their overall risk rating."""

        if detection_result.image != safety_result.image:
            raise ValueError("object detection and safety detection image dimensions must match")

        events = [self._assess_event(event) for event in safety_result.events]
        overall_score = max((event.risk.score for event in events), default=0.0)

        return AnalysisResult(
            image=detection_result.image,
            overall_risk=self._rating(overall_score),
            detected_objects=detection_result.detections,
            events=events,
        )

    def _assess_event(self, event: SafetyEvent) -> RiskEvent:
        if event.rule_type == _MISSING_PPE_RULE:
            return self._assess_missing_ppe(event)
        if event.rule_type == _PERSON_NEAR_FORKLIFT_RULE:
            return self._assess_person_near_forklift(event)
        raise ValueError(f"unsupported safety rule type: {event.rule_type}")

    def _assess_missing_ppe(self, event: SafetyEvent) -> RiskEvent:
        missing_ppe = list(dict.fromkeys(_require_string_list(event, "missing_ppe")))
        if not missing_ppe:
            raise ValueError("missing_required_ppe event must contain at least one missing PPE item")

        individual_scores = {
            "safety_hat": self._config.missing_ppe.missing_safety_hat_score,
            "safety_vest": self._config.missing_ppe.missing_safety_vest_score,
        }
        unsupported = [item for item in missing_ppe if item not in individual_scores]
        if unsupported:
            raise ValueError(f"unsupported missing PPE types: {', '.join(unsupported)}")

        score = (
            self._config.missing_ppe.multiple_missing_ppe_score
            if len(missing_ppe) > 1
            else individual_scores[missing_ppe[0]]
        )
        return RiskEvent(
            rule_type=event.rule_type,
            risk=self._rating(score),
            subject=event.subject,
            related_objects=event.related_objects,
            attributes=event.attributes,
        )

    def _assess_person_near_forklift(self, event: SafetyEvent) -> RiskEvent:
        return RiskEvent(
            rule_type=event.rule_type,
            risk=self._rating(self._config.person_forklift_proximity.score),
            subject=event.subject,
            related_objects=event.related_objects,
            attributes=event.attributes,
        )

    def _rating(self, score: float) -> RiskRating:
        thresholds = self._config.risk_level_thresholds
        if score >= thresholds.high_min_score:
            level = RiskLevel.HIGH
        elif score >= thresholds.medium_min_score:
            level = RiskLevel.MEDIUM
        else:
            level = RiskLevel.LOW
        return RiskRating(score=score, level=level)


def _require_string_list(event: SafetyEvent, attribute: str) -> list[str]:
    value = event.attributes.get(attribute)
    if not isinstance(value, list):
        raise ValueError(f"{event.rule_type} event attribute {attribute!r} must be a list of strings")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{event.rule_type} event attribute {attribute!r} must be a list of strings")
        items.append(item)
    return items
