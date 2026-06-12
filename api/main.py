import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from api.model_loader import load_model, run_prediction
from api.schemas import HealthResponse, PredictResponse, WaterPotabilityFeatures

logger = logging.getLogger(__name__)

models: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Ładuje model predykcyjny przy starcie aplikacji i zwalnia go przy zamknięciu."""
    models.update(load_model())
    yield
    models.clear()


app = FastAPI(
    title="Water Potability Prediction API",
    description="REST API serwujące predykcje jakości wody (Potability 0/1).",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Zwraca status API oraz informację o załadowanym modelu."""
    return HealthResponse(
        status="ok",
        model_loaded=bool(models.get("model_loaded")),
        model_name=models.get("model_name"),
        model_type=models.get("model_type"),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(features: WaterPotabilityFeatures) -> PredictResponse:
    """Wykonuje predykcję na podstawie zwalidowanych cech jakości wody."""
    if not models.get("model_loaded"):
        raise HTTPException(status_code=503, detail="Model is not loaded")

    predictor = models.get("predictor")
    model_name = models.get("model_name")

    try:
        prediction = run_prediction(predictor, features.model_dump())
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictResponse(prediction=prediction, model=model_name)
