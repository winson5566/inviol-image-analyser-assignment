from functools import cache
from io import BytesIO
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from inviol_image_analyser_assignment.config import AnalysisConfig, load_analysis_config
from inviol_image_analyser_assignment.models import AnalysisResult, ObjectDetectionResult, SafetyDetectionResult
from inviol_image_analyser_assignment.services.object_detection import ObjectDetector, create_object_detector
from inviol_image_analyser_assignment.services.safety_detection import SafetyRuleEngine, create_safety_rule_engine

app = FastAPI()


@app.get("/healthcheck")
async def get_healthcheck():
    return {"status": "healthy"}


@app.post("/analyse")
async def post_analyse(file: UploadFile) -> AnalysisResult:
    # TODO: Implement the actual image analysis logic
    print(f"Received file: {file.filename}")
    return AnalysisResult(risk_rating=5)


@cache
def _get_analysis_config() -> AnalysisConfig:
    """Load and cache the active analysis configuration."""

    config_path = Path(__file__).resolve().parents[2] / "config" / "analysis.json"
    return load_analysis_config(config_path)


@cache
def get_object_detector() -> ObjectDetector:
    """Build and cache the detector selected by the active configuration."""

    return create_object_detector(_get_analysis_config().object_detection)


@cache
def get_safety_rule_engine() -> SafetyRuleEngine:
    """Build and cache the safety rules selected by the active configuration."""

    return create_safety_rule_engine(_get_analysis_config().safety_rules)


def _decode_image(content: bytes) -> Image.Image:
    """Decode uploaded image bytes into the RGB format expected by detectors."""

    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded image is empty")
    try:
        with Image.open(BytesIO(content)) as image:
            image.load()
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is not a valid image"
        ) from error


@app.post("/object-detection")
async def post_object_detection(
    file: UploadFile,
    detector: Annotated[ObjectDetector, Depends(get_object_detector)],
) -> ObjectDetectionResult:
    """Detect configured workplace objects in an uploaded image."""

    image = _decode_image(await file.read())
    return await run_in_threadpool(detector.detect, image)


@app.post("/safety-detection")
async def post_safety_detection(
    file: UploadFile,
    detector: Annotated[ObjectDetector, Depends(get_object_detector)],
    rule_engine: Annotated[SafetyRuleEngine, Depends(get_safety_rule_engine)],
) -> SafetyDetectionResult:
    """Detect workplace objects and evaluate the configured safety rules."""

    image = _decode_image(await file.read())
    detection_result = await run_in_threadpool(detector.detect, image)
    return await run_in_threadpool(rule_engine.evaluate, detection_result)
