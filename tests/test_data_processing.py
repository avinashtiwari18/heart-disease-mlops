"""Unit tests for data loading and cleaning."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_processing import (  # noqa: E402
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    clean_dataframe,
    load_raw_dataframe,
    missing_value_report,
    train_test_split_data,
)


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    """Prefer real cleaned data; fall back to synthetic rows."""
    cleaned = ROOT / "data" / "cleaned" / "heart_disease_cleaned.csv"
    if cleaned.exists():
        return load_raw_dataframe(cleaned)

    rng = np.random.default_rng(42)
    n = 40
    data = {
        "age": rng.integers(30, 80, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(0, 4, n),
        "trestbps": rng.integers(100, 180, n),
        "chol": rng.integers(150, 350, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(90, 190, n),
        "exang": rng.integers(0, 2, n),
        "oldpeak": rng.random(n) * 4,
        "slope": rng.integers(0, 3, n),
        "ca": rng.integers(0, 4, n),
        "thal": rng.integers(0, 4, n),
        "target": rng.integers(0, 2, n),
    }
    return pd.DataFrame(data)


def test_feature_columns_present(sample_df):
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        assert col in sample_df.columns


def test_clean_dataframe_no_missing(sample_df):
    dirty = sample_df.copy()
    dirty.loc[0, "chol"] = np.nan
    dirty.loc[1, "ca"] = np.nan
    cleaned = clean_dataframe(dirty)
    assert cleaned.isna().sum().sum() == 0
    assert cleaned.shape[0] == dirty.shape[0]


def test_missing_value_report_shape(sample_df):
    report = missing_value_report(sample_df)
    assert "missing_count" in report.columns
    assert len(report) == sample_df.shape[1]


def test_train_test_split_stratified(sample_df):
    cleaned = clean_dataframe(sample_df)
    # Ensure both classes exist
    if cleaned[TARGET_COLUMN].nunique() < 2:
        cleaned.loc[:1, TARGET_COLUMN] = [0, 1]
    x_train, x_test, y_train, y_test = train_test_split_data(cleaned, test_size=0.25)
    assert len(x_train) + len(x_test) == len(cleaned)
    assert set(y_train.unique()).issubset({0, 1})
    assert list(x_train.columns) == FEATURE_COLUMNS


def test_outlier_clipping():
    df = pd.DataFrame(
        [
            {
                "age": 55,
                "sex": 1,
                "cp": 1,
                "trestbps": 300,
                "chol": 900,
                "fbs": 0,
                "restecg": 0,
                "thalach": 300,
                "exang": 0,
                "oldpeak": 10,
                "slope": 1,
                "ca": 0,
                "thal": 2,
                "target": 1,
            }
        ]
    )
    cleaned = clean_dataframe(df)
    assert cleaned.loc[0, "chol"] <= 500
    assert cleaned.loc[0, "trestbps"] <= 220
    assert cleaned.loc[0, "thalach"] <= 220
    assert cleaned.loc[0, "oldpeak"] <= 6.5
