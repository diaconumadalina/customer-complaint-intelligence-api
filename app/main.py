from fastapi import FastAPI, Depends
from app.schemas import PredictRequest, PredictResponse
from app.dependencies import get_pipeline
from model.inference import ComplaintInferencePipeline

app = FastAPI(
    title="Customer Complaint Intelligence API",
    version="1.0.0",
    description="A production-grade API that classifies financial complaints using SBERT + PyTorch."
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(
        payload: PredictRequest,
        pipeline: ComplaintInferencePipeline = Depends(get_pipeline)
):
    label, probs = pipeline.predict(payload.text)
    return PredictResponse(label=label, probabilities=probs)
