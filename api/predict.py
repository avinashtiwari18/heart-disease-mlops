"""Model loading and inference helpers for the FastAPI service."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd

from src.data_processing import FEATURE_COLUMNS

logger = logging.getLogger("heart_disease_api")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "heart_disease_pipeline.joblib"
DEFAULT_META_PATH = ROOT / "models" / "model_metadata.json"

_model = None
_metadata: Dict[str, Any] = {}


def get_model_path() -> Path:
    return Path(__import__("os").environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))


def load_metadata() -> Dict[str, Any]:
    global _metadata
    meta_path = DEFAULT_META_PATH
    if meta_path.exists():
        _metadata = json.loads(meta_path.read_text())
    else:
        _metadata = {"best_model": "unknown"}
    return _metadata


def load_model(force: bool = False):
    global _model
    if _model is not None and not force:
        return _model

    model_path = get_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train first: python -m src.train"
        )
    _model = joblib.load(model_path)
    load_metadata()
    logger.info("Loaded model from %s (%s)", model_path, _metadata.get("best_model"))
    return _model


def features_to_dataframe(features: Dict[str, Any]) -> pd.DataFrame:
    row = {col: features[col] for col in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_one(features: Dict[str, Any]) -> Tuple[int, float, float, str]:
    """
    Returns:
        prediction, confidence (pred class), probability_disease, model_name
    """
    model = load_model()
    frame = features_to_dataframe(features)
    proba = model.predict_proba(frame)[0]
    pred = int(model.predict(frame)[0])
    confidence = float(proba[pred])
    prob_disease = float(proba[1])
    model_name = str(_metadata.get("best_model", "unknown"))
    return pred, confidence, prob_disease, model_name


def model_info() -> Dict[str, Optional[str]]:
    load_metadata()
    loaded = _model is not None or get_model_path().exists()
    return {
        "model_loaded": loaded,
        "model_name": _metadata.get("best_model"),
        "model_path": str(get_model_path()),
    }
