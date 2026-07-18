"""
FastAPI service for Heart Disease risk prediction.

Endpoints:
  GET  /health   - liveness / model status
  POST /predict  - single patient prediction
  GET  /metrics  - Prometheus metrics
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from prometheus_fastapi_instrumentator import Instrumentator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.predict import load_model, model_info, predict_one  # noqa: E402
from api.schemas import HealthResponse, PatientFeatures, PredictionResponse  # noqa: E402

LOG_DIR = ROOT / "artifacts" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "api_requests.log"),
    ],
)
logger = logging.getLogger("heart_disease_api")

app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description=(
        "MLOps Assignment 01 – AIMLCZG523. "
        "Predicts heart-disease risk from UCI clinical features."
    ),
    version="1.0.0",
)

# Prometheus metrics at /metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=True)


@app.on_event("startup")
def startup_event() -> None:
    try:
        load_model()
        logger.info("Model loaded successfully on startup")
    except Exception as exc:  # noqa: BLE001
        logger.error("Model load failed on startup: %s", exc)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/", tags=["meta"])
def root():
    return {
        "service": "heart-disease-risk-api",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "metrics": "/metrics",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    info = model_info()
    status = "ok" if info["model_loaded"] else "degraded"
    return HealthResponse(
        status=status,
        model_loaded=bool(info["model_loaded"]),
        model_name=info.get("model_name"),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(payload: PatientFeatures):
    try:
        pred, confidence, prob_disease, model_name = predict_one(payload.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    label = "heart_disease_risk" if pred == 1 else "low_risk"
    logger.info(
        "prediction=%s confidence=%.4f prob_disease=%.4f model=%s",
        pred,
        confidence,
        prob_disease,
        model_name,
    )
    return PredictionResponse(
        prediction=pred,
        prediction_label=label,
        confidence=round(confidence, 4),
        probability_disease=round(prob_disease, 4),
        model_name=model_name,
    )
