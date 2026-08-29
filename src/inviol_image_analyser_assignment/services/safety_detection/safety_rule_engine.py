"""Model-independent safety-rule evaluation framework."""

from collections.abc import Sequence
from typing import Protocol, final

from inviol_image_analyser_assignment.config import SafetyRulesConfig
from inviol_image_analyser_assignment.models import ObjectDetectionResult, SafetyDetectionResult, SafetyEvent
from inviol_image_analyser_assignment.services.safety_detection.missing_ppe import MissingPpeRule


class SafetyRule(Protocol):
    """Contract implemented by every workplace safety rule."""

    rule_type: str

    def evaluate(self, detection_result: ObjectDetectionResult) -> Sequence[SafetyEvent]:
        """Return every safety event produced by this rule."""

        ...


@final
class SafetyRuleEngine:
    """Evaluate independent safety rules in a deterministic order."""

    def __init__(self, rules: Sequence[SafetyRule]) -> None:
        rule_types = [rule.rule_type for rule in rules]
        if any(not rule_type for rule_type in rule_types):
            raise ValueError("safety rule types must not be empty")
        if len(rule_types) != len(set(rule_types)):
            raise ValueError("safety rule types must be unique")
        self._rules: tuple[SafetyRule, ...] = tuple(rules)

    def evaluate(self, detection_result: ObjectDetectionResult) -> SafetyDetectionResult:
        """Run all registered rules and combine their events."""

        events: list[SafetyEvent] = []
        for rule in self._rules:
            rule_events = rule.evaluate(detection_result)
            if any(event.rule_type != rule.rule_type for event in rule_events):
                raise ValueError(f"safety rule {rule.rule_type!r} returned an event for another rule type")
            events.extend(rule_events)

        return SafetyDetectionResult(image=detection_result.image, events=events)


def create_safety_rule_engine(config: SafetyRulesConfig) -> SafetyRuleEngine:
    """Build an engine containing all configured safety rules."""

    return SafetyRuleEngine([MissingPpeRule(config.missing_ppe)])
