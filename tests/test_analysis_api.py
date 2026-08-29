"""API test for the complete image-analysis pipeline."""

from io import BytesIO
from typing import final

from fastapi.testclient import TestClient
from PIL import Image

from inviol_image_analyser_assignment.app import (
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
)


@final
class FakeObjectDetector:
    def __init__(self, result: ObjectDetectionResult) -> None:
        self._result = result
        self.received_images: list[tuple[str | None, tuple[int, int]]] = []

    def detect(self, image: Image.Image) -> ObjectDetectionResult:
        self.received_images.append((image.mode, image.size))
        return self._result


def test_analyse_returns_structured_risk_assessment() -> None:
    person = Detection(
        object_type=ObjectType.PERSON,
        confidence=0.9,
        bounding_box=BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.9),
    )
    detector = FakeObjectDetector(
        ObjectDetectionResult(
            image=ImageDimensions(width=20, height=10),
            detections=[person],
            latency_ms=5,
        )
    )
    image_bytes = BytesIO()
    Image.new("RGB", (20, 10), "white").save(image_bytes, format="PNG")
    app.dependency_overrides[get_object_detector] = lambda: detector
    get_safety_rule_engine.cache_clear()
    get_risk_assessment_service.cache_clear()

    try:
        with TestClient(app) as client:
            response = client.post(
                "/analyse",
                files={"file": ("workplace.png", image_bytes.getvalue(), "image/png")},
            )
    finally:
        app.dependency_overrides.pop(get_object_detector, None)
        get_safety_rule_engine.cache_clear()
        get_risk_assessment_service.cache_clear()

    result = AnalysisResult.model_validate_json(response.content)
    assert response.status_code == 200
    assert detector.received_images == [("RGB", (20, 10))]
    assert result.image == ImageDimensions(width=20, height=10)
    assert result.detected_objects == [person]
    assert len(result.events) == 1
    assert result.overall_risk == result.events[0].risk
    assert result.events[0].rule_type == "missing_required_ppe"
