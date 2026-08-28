from fastapi import FastAPI, UploadFile

from inviol_image_analyser_assignment.models import AnalysisResult

app = FastAPI()


@app.get("/healthcheck")
async def get_healthcheck():
    return {"status": "healthy"}


@app.post("/analyse")
async def post_analyse(file: UploadFile) -> AnalysisResult:
    # TODO: Implement the actual image analysis logic
    print(f"Received file: {file.filename}")
    return AnalysisResult(risk_rating=5)
