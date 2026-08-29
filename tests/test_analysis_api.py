"""API test for the complete image-analysis pipeline."""

from io import BytesIO
from typing import final

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from PIL import Image

from inviol_image_analyser_assignment.app import (
    MAX_IMAGE_UPLOAD_BYTES,
    app,
    get_object_detector,
    get_risk_assessment_service,
    get_safety_rule_engine,
)
from inviol_image_analyser_assignment.models import (
    AnalysisResult,
    BoundingBox,
    Detection,
    ImageDimensions,
    ObjectDetectionResult,
    ObjectType,
    RiskLevel,
    RiskRating,
)


@final
class FakeObjectDetector:
    def __init__(self, result: ObjectDetectionResult) -> None:
        self._result = result
        self.received_images: list[tuple[str | None, tuple[int, int]]] = []

    def detect(self, image: Image.Image) -> ObjectDetectionResult:
        self.received_images.append((image.mode, image.size))
        return self._result


def _post_image(detector: FakeObjectDetector, content: bytes, content_type: str) -> Response:
    app.dependency_overrides[get_object_detector] = lambda: detector
    get_safety_rule_engine.cache_clear()
    get_risk_assessment_service.cache_clear()

    try:
        with TestClient(app) as client:
            return client.post(
                "/analyse",
                files={"file": ("workplace", content, content_type)},
            )
    finally:
        app.dependency_overrides.pop(get_object_detector, None)
        get_safety_rule_engine.cache_clear()
        get_risk_assessment_service.cache_clear()


def _empty_detector() -> FakeObjectDetector:
    return FakeObjectDetector(
        ObjectDetectionResult(
            image=ImageDimensions(width=1, height=1),
            detections=[],
            latency_ms=0,
        )
    )


def _detection(object_type: ObjectType, box: tuple[float, float, float, float], confidence: float = 0.9) -> Detection:
    return Detection(
        object_type=object_type,
        confidence=confidence,
        bounding_box=BoundingBox(x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]),
    )


def test_analyse_returns_structured_risk_assessment() -> None:
    person = _detection(ObjectType.PERSON, (0.1, 0.1, 0.5, 0.9))
    detector = FakeObjectDetector(
        ObjectDetectionResult(
            image=ImageDimensions(width=20, height=10),
            detections=[person],
            latency_ms=5,
        )
    )
    image_bytes = BytesIO()
    Image.new("RGB", (20, 10), "white").save(image_bytes, format="PNG")
    response = _post_image(detector, image_bytes.getvalue(), "image/png")

    result = AnalysisResult.model_validate_json(response.content)
    assert response.status_code == 200
    assert detector.received_images == [("RGB", (20, 10))]
    assert result.image == ImageDimensions(width=20, height=10)
    assert result.detected_objects == [person]
    assert len(result.events) == 1
    assert result.overall_risk == result.events[0].risk
    assert result.events[0].rule_type == "missing_required_ppe"


def test_analyse_returns_zero_risk_when_no_objects_are_detected() -> None:
    image_bytes = BytesIO()
    Image.new("RGB", (1, 1), "white").save(image_bytes, format="PNG")
    detector = _empty_detector()

    response = _post_image(detector, image_bytes.getvalue(), "image/png")

    result = AnalysisResult.model_validate_json(response.content)
    assert response.status_code == 200
    assert result.detected_objects == []
    assert result.events == []
    assert result.overall_risk == RiskRating(score=0, level=RiskLevel.LOW)


def test_analyse_returns_zero_risk_when_detected_person_has_required_ppe() -> None:
    detections = [
        _detection(ObjectType.PERSON, (0.1, 0.1, 0.5, 0.9)),
        _detection(ObjectType.SAFETY_HAT, (0.2, 0.12, 0.3, 0.25), confidence=0.8),
        _detection(ObjectType.SAFETY_VEST, (0.18, 0.35, 0.42, 0.7), confidence=0.85),
    ]
    detector = FakeObjectDetector(
        ObjectDetectionResult(
            image=ImageDimensions(width=20, height=10),
            detections=detections,
            latency_ms=5,
        )
    )
    image_bytes = BytesIO()
    Image.new("RGB", (20, 10), "white").save(image_bytes, format="PNG")

    response = _post_image(detector, image_bytes.getvalue(), "image/png")

    result = AnalysisResult.model_validate_json(response.content)
    assert response.status_code == 200
    assert result.detected_objects == detections
    assert result.events == []
    assert result.overall_risk == RiskRating(score=0, level=RiskLevel.LOW)


@pytest.mark.parametrize(
    ("content", "content_type", "expected_status", "expected_detail"),
    [
        (b"", "image/png", 400, "Uploaded image is empty"),
        (b"not an image", "image/png", 400, "Uploaded file is not a valid image"),
        (b"not an image", "application/octet-stream", 415, "Only JPEG and PNG images are supported"),
    ],
)
def test_analyse_rejects_invalid_uploads(
    content: bytes,
    content_type: str,
    expected_status: int,
    expected_detail: str,
) -> None:
    response = _post_image(_empty_detector(), content, content_type)

    assert response.status_code == expected_status
    assert response.json() == {"detail": expected_detail}


def test_analyse_rejects_unsupported_image_format() -> None:
    image_bytes = BytesIO()
    Image.new("RGB", (1, 1), "white").save(image_bytes, format="GIF")

    response = _post_image(_empty_detector(), image_bytes.getvalue(), "image/png")

    assert response.status_code == 415
    assert response.json() == {"detail": "Only JPEG and PNG images are supported"}


def test_analyse_rejects_oversized_upload() -> None:
    response = _post_image(_empty_detector(), b"x" * (MAX_IMAGE_UPLOAD_BYTES + 1), "image/png")

    assert response.status_code == 413
    assert response.json() == {"detail": "Uploaded image must not exceed 10 MB"}
