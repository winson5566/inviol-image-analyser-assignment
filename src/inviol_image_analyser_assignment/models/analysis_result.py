from pydantic import BaseModel


class AnalysisResult(BaseModel):
    risk_rating: int
    # TODO: Add any additional fields you deem necessary
