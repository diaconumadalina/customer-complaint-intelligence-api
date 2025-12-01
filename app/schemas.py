from pydantic import BaseModel, Field
from typing import Dict


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=5,
        description="Customer complaint text to classify"
    )


class PredictResponse(BaseModel):
    label: str
    probabilities: Dict[str, float]
