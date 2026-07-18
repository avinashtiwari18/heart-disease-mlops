"""Unit tests for feature pipeline and trained model inference."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_processing import FEATURE_COLUMNS, clean_dataframe  # noqa: E402
from src.features import build_model_pipeline, build_preprocessor  # noqa: E402


@pytest.fixture
def tiny_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 30
    return pd.DataFrame(
        {
            "age": rng.integers(40, 70, n),
            "sex": rng.integers(0, 2, n),
            "cp": rng.integers(0, 4, n),
            "trestbps": rng.integers(110, 170, n),
            "chol": rng.integers(180, 320, n),
            "fbs": rng.integers(0, 2, n),
            "restecg": rng.integers(0, 3, n),
            "thalach": rng.integers(100, 180, n),
            "exang": rng.integers(0, 2, n),
            "oldpeak": rng.random(n) * 3,
            "slope": rng.integers(0, 3, n),
            "ca": rng.integers(0, 4, n),
            "thal": rng.integers(0, 4, n),
            "target": np.array([0, 1] * (n // 2)),
        }
    )


def test_preprocessor_output_shape(tiny_frame):
    pre = build_preprocessor()
    x = tiny_frame[FEATURE_COLUMNS]
    transformed = pre.fit_transform(x)
    assert transformed.shape[0] == len(x)
    assert transformed.shape[1] > len(FEATURE_COLUMNS)  # one-hot expands cats


def test_model_pipeline_fit_predict(tiny_frame):
    df = clean_dataframe(tiny_frame)
    x = df[FEATURE_COLUMNS]
    y = df["target"]
    pipe = build_model_pipeline(LogisticRegression(max_iter=1000, random_state=42))
    pipe.fit(x, y)
    preds = pipe.predict(x)
    proba = pipe.predict_proba(x)
    assert len(preds) == len(y)
    assert proba.shape == (len(y), 2)
    assert set(np.unique(preds)).issubset({0, 1})


def test_saved_model_if_present():
    model_path = ROOT / "models" / "heart_disease_pipeline.joblib"
    if not model_path.exists():
        pytest.skip("Trained model not available yet")

    model = joblib.load(model_path)
    sample = pd.DataFrame(
        [
            {
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
        ],
        columns=FEATURE_COLUMNS,
    )
    pred = model.predict(sample)[0]
    proba = model.predict_proba(sample)[0]
    assert pred in (0, 1)
    assert np.isclose(proba.sum(), 1.0, atol=1e-5)
