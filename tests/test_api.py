"""API contract tests for /health and /predict."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "models" / "heart_disease_pipeline.joblib"


@pytest.fixture(scope="module")
def client():
    if not MODEL_PATH.exists():
        pytest.skip("Trained model required for API tests")
    from api.main import app

    with TestClient(app) as c:
        yield c


SAMPLE_PAYLOAD = {
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1,
}


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "predict" in response.json()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert "model_loaded" in body


def test_predict_success(client):
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["confidence"] <= 1.0
    assert 0.0 <= body["probability_disease"] <= 1.0
    assert body["prediction_label"] in {"low_risk", "heart_disease_risk"}


def test_predict_validation_error(client):
    bad = dict(SAMPLE_PAYLOAD)
    bad["sex"] = 5  # invalid
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests" in response.text or "python_" in response.text
