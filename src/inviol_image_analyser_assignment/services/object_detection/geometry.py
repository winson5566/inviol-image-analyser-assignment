"""Model-independent bounding-box normalization and suppression."""

from collections.abc import Sequence

from inviol_image_analyser_assignment.models import BoundingBox, Detection


def normalized_bounding_box(coordinates: Sequence[float]) -> BoundingBox | None:
    """Clamp normalized coordinates and discard malformed or degenerate boxes."""

    if len(coordinates) != 4:
        return None
    x_min, y_min, x_max, y_max = (max(0.0, min(float(value), 1.0)) for value in coordinates)
    if x_min >= x_max or y_min >= y_max:
        return None
    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def bounding_box_overlap_fraction(subject: BoundingBox, container: BoundingBox) -> float:
    """Return the fraction of a subject box contained by another box."""

    intersection_width = max(0.0, min(subject.x_max, container.x_max) - max(subject.x_min, container.x_min))
    intersection_height = max(0.0, min(subject.y_max, container.y_max) - max(subject.y_min, container.y_min))
    intersection = intersection_width * intersection_height
    subject_area = (subject.x_max - subject.x_min) * (subject.y_max - subject.y_min)
    return intersection / subject_area


def class_aware_nms(detections: Sequence[Detection], iou_threshold: float) -> list[Detection]:
    """Suppress lower-confidence overlapping detections of the same object type."""

    kept: list[Detection] = []
    for candidate in sorted(detections, key=lambda detection: detection.confidence, reverse=True):
        if not any(
            candidate.object_type is existing.object_type
            and _intersection_over_union(candidate.bounding_box, existing.bounding_box) > iou_threshold
            for existing in kept
        ):
            kept.append(candidate)
    return kept


def _intersection_over_union(left: BoundingBox, right: BoundingBox) -> float:
    """Calculate intersection-over-union for two normalized bounding boxes."""

    intersection_width = max(0.0, min(left.x_max, right.x_max) - max(left.x_min, right.x_min))
    intersection_height = max(0.0, min(left.y_max, right.y_max) - max(left.y_min, right.y_min))
    intersection = intersection_width * intersection_height
    left_area = (left.x_max - left.x_min) * (left.y_max - left.y_min)
    right_area = (right.x_max - right.x_min) * (right.y_max - right.y_min)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0
