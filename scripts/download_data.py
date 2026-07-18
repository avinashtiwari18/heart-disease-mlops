#!/usr/bin/env python3
"""
Download and prepare the Heart Disease UCI dataset.

Primary source: UCI ML Repository (processed Cleveland subset).
Fallback: public mirror of the same processed Cleveland file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

COLUMNS = [
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
    "num",
]

# Cleveland processed file (14 attributes). Missing values encoded as '?'
UCI_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/heart.csv",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def download_bytes(url: str, timeout: int = 60) -> bytes:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def load_uci_cleveland(raw_path: Path) -> pd.DataFrame:
    """Load Cleveland file (no header, '?' missing)."""
    df = pd.read_csv(raw_path, header=None, names=COLUMNS, na_values="?")
    # Binary target: 0 = no disease, 1 = disease (any num > 0)
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])
    return df


def load_jbrownlee_heart(raw_path: Path) -> pd.DataFrame:
    """Load jbrownlee heart.csv (already binary target)."""
    df = pd.read_csv(raw_path)
    # Normalize column names if needed
    rename = {
        "age": "age",
        "sex": "sex",
        "cp": "cp",
        "trestbps": "trestbps",
        "chol": "chol",
        "fbs": "fbs",
        "restecg": "restecg",
        "thalach": "thalach",
        "exang": "exang",
        "oldpeak": "oldpeak",
        "slope": "slope",
        "ca": "ca",
        "thal": "thal",
        "target": "target",
    }
    df = df.rename(columns={c: rename.get(c, c) for c in df.columns})
    expected = [c for c in COLUMNS if c != "num"] + ["target"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Unexpected schema, missing columns: {missing}")
    return df[expected]


def download_dataset(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "heart_disease_raw.csv"
    cleaned_path = output_dir.parent / "cleaned" / "heart_disease_cleaned.csv"
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)

    last_error: Exception | None = None
    for url in UCI_URLS:
        try:
            print(f"Downloading from: {url}")
            content = download_bytes(url)
            raw_path.write_bytes(content)

            if "processed.cleveland" in url:
                df = load_uci_cleveland(raw_path)
            else:
                df = load_jbrownlee_heart(raw_path)

            # Light clean for the cleaned deliverable (median impute for EDA-ready CSV)
            for col in df.columns:
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].median())

            df.to_csv(cleaned_path, index=False)
            print(f"Saved raw file  -> {raw_path}")
            print(f"Saved cleaned   -> {cleaned_path}")
            print(f"Shape: {df.shape}, target balance:\n{df['target'].value_counts()}")
            return cleaned_path
        except Exception as exc:  # noqa: BLE001 - try next mirror
            print(f"Failed ({url}): {exc}", file=sys.stderr)
            last_error = exc

    raise RuntimeError(f"Could not download dataset from any source: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Heart Disease UCI dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root() / "data" / "raw",
        help="Directory for raw download",
    )
    args = parser.parse_args()
    download_dataset(args.output_dir)


if __name__ == "__main__":
    main()
