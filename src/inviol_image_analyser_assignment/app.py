from functools import cache
from io import BytesIO
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from inviol_image_analyser_assignment.config import load_analysis_config
from inviol_image_analyser_assignment.models import AnalysisResult, ObjectDetectionResult
from inviol_image_analyser_assignment.services.object_detection import ObjectDetector, create_object_detector

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
def _get_object_detector() -> ObjectDetector:
    """Build and cache the detector selected by the active configuration."""

    config_path = Path(__file__).resolve().parents[2] / "config" / "analysis.json"
    config = load_analysis_config(config_path)
    return create_object_detector(config.object_detection)


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
    detector: Annotated[ObjectDetector, Depends(_get_object_detector)],
) -> ObjectDetectionResult:
    """Detect configured workplace objects in an uploaded image."""

    image = _decode_image(await file.read())
    return await run_in_threadpool(detector.detect, image)
