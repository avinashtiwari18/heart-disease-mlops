"""Data loading, cleaning, and train/test split utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
TARGET_COLUMN = "target"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_path() -> Path:
    cleaned = project_root() / "data" / "cleaned" / "heart_disease_cleaned.csv"
    if cleaned.exists():
        return cleaned
    raw = project_root() / "data" / "raw" / "heart_disease_raw.csv"
    return raw


def load_raw_dataframe(path: Path | None = None) -> pd.DataFrame:
    """Load heart disease CSV and normalize schema."""
    path = path or default_data_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: python scripts/download_data.py"
        )

    df = pd.read_csv(path)

    # Cleveland raw (no header / num column)
    if list(df.columns) == list(range(len(df.columns))) or "num" in df.columns:
        columns = FEATURE_COLUMNS + ["num"]
        if "num" not in df.columns:
            df = pd.read_csv(path, header=None, names=columns, na_values="?")
        else:
            df = df.replace("?", np.nan)
            if set(FEATURE_COLUMNS).issubset(df.columns):
                df["num"] = pd.to_numeric(df["num"], errors="coerce")
        df["target"] = (pd.to_numeric(df["num"], errors="coerce") > 0).astype(int)
        df = df.drop(columns=["num"], errors="ignore")

    # Coerce feature types
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if TARGET_COLUMN not in df.columns:
        raise ValueError("Dataset must contain a 'target' column")

    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").astype(int)
    return df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    pct = (missing / len(df) * 100).round(2)
    return pd.DataFrame({"missing_count": missing, "missing_pct": pct})


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset for modelling:
    - coerce numerics
    - impute missing values with median (per column)
    - clip extreme cholesterol/blood-pressure outliers lightly
    """
    cleaned = df.copy()
    for col in FEATURE_COLUMNS:
        if cleaned[col].isna().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    # Soft outlier winsorization for clinical ranges
    cleaned["chol"] = cleaned["chol"].clip(lower=100, upper=500)
    cleaned["trestbps"] = cleaned["trestbps"].clip(lower=80, upper=220)
    cleaned["thalach"] = cleaned["thalach"].clip(lower=60, upper=220)
    cleaned["oldpeak"] = cleaned["oldpeak"].clip(lower=0, upper=6.5)

    # Ensure categorical codes are integers
    for col in CATEGORICAL_FEATURES:
        cleaned[col] = cleaned[col].round().astype(int)

    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)
    return cleaned


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    x = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return x, y


def train_test_split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x, y = split_features_target(df)
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def save_processed(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or (project_root() / "data" / "processed" / "heart_disease_processed.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
