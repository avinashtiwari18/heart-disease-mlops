#!/usr/bin/env python3
"""
Train and compare heart-disease classifiers with MLflow tracking.

Models:
  1) Logistic Regression (baseline, interpretable)
  2) Random Forest (non-linear ensemble)
  3) XGBoost (optional strong booster)

The best model (by ROC-AUC on held-out test set) is saved with its
full preprocessing pipeline via joblib + MLflow.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:  # noqa: BLE001 - OpenMP missing on some macOS setups
    XGBClassifier = None  # type: ignore[assignment,misc]
    HAS_XGBOOST = False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_processing import (  # noqa: E402
    clean_dataframe,
    load_raw_dataframe,
    save_processed,
    train_test_split_data,
)
from src.features import build_model_pipeline  # noqa: E402


def ensure_dirs() -> Dict[str, Path]:
    paths = {
        "models": ROOT / "models",
        "artifacts": ROOT / "artifacts" / "metrics",
        "eda": ROOT / "artifacts" / "eda",
        "mlflow": ROOT / "mlflow_tracking",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def compute_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def plot_confusion_matrix(y_true, y_pred, title: str, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc(y_true, y_prob, title: str, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def get_model_specs() -> List[Tuple[str, Any, Dict[str, Any]]]:
    """Return (name, estimator, param_grid) triples."""
    specs: List[Tuple[str, Any, Dict[str, Any]]] = [
        (
            "logistic_regression",
            LogisticRegression(max_iter=2000, random_state=42),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__penalty": ["l2"],
                "model__solver": ["lbfgs"],
            },
        ),
        (
            "random_forest",
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_split": [2, 5],
            },
        ),
    ]
    if HAS_XGBOOST:
        specs.append(
            (
                "xgboost",
                XGBClassifier(
                    random_state=42,
                    eval_metric="logloss",
                    n_jobs=-1,
                    verbosity=0,
                ),
                {
                    "model__n_estimators": [100, 200],
                    "model__max_depth": [3, 5],
                    "model__learning_rate": [0.05, 0.1],
                },
            )
        )
    else:
        # Fallback booster when XGBoost native libs (libomp) are unavailable
        specs.append(
            (
                "gradient_boosting",
                GradientBoostingClassifier(random_state=42),
                {
                    "model__n_estimators": [100, 200],
                    "model__max_depth": [2, 3],
                    "model__learning_rate": [0.05, 0.1],
                },
            )
        )
    return specs


def train_and_evaluate(
    experiment_name: str = "heart-disease-uci",
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
) -> Dict[str, Any]:
    paths = ensure_dirs()
    mlflow.set_tracking_uri(paths["mlflow"].as_uri())
    mlflow.set_experiment(experiment_name)

    df_raw = load_raw_dataframe()
    df = clean_dataframe(df_raw)
    save_processed(df)

    x_train, x_test, y_train, y_test = train_test_split_data(
        df, test_size=test_size, random_state=random_state
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    leaderboard: List[Dict[str, Any]] = []
    best_name = None
    best_auc = -1.0
    best_pipeline = None
    best_params: Dict[str, Any] = {}

    for name, estimator, param_grid in get_model_specs():
        pipeline = build_model_pipeline(estimator)
        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )

        with mlflow.start_run(run_name=name):
            search.fit(x_train, y_train)
            model = search.best_estimator_

            y_pred = model.predict(x_test)
            y_prob = model.predict_proba(x_test)[:, 1]
            metrics = compute_metrics(y_test, y_pred, y_prob)

            cv_scores = cross_val_score(
                model, x_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
            )
            metrics["cv_roc_auc_mean"] = float(cv_scores.mean())
            metrics["cv_roc_auc_std"] = float(cv_scores.std())

            # Log params / metrics
            mlflow.log_params({f"best_{k}": v for k, v in search.best_params_.items()})
            mlflow.log_param("model_name", name)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("random_state", random_state)
            mlflow.log_metrics(metrics)

            cm_path = paths["artifacts"] / f"{name}_confusion_matrix.png"
            roc_path = paths["artifacts"] / f"{name}_roc_curve.png"
            plot_confusion_matrix(
                y_test, y_pred, f"{name} Confusion Matrix", cm_path
            )
            plot_roc(y_test, y_prob, f"{name} ROC Curve", roc_path)
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(roc_path))

            mlflow.sklearn.log_model(model, artifact_path="model")

            row = {"model": name, **metrics, "best_params": search.best_params_}
            leaderboard.append(row)
            print(f"\n=== {name} ===")
            print(json.dumps({k: v for k, v in row.items() if k != "best_params"}, indent=2))
            print("best_params:", search.best_params_)

            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_name = name
                best_pipeline = model
                best_params = search.best_params_

    assert best_pipeline is not None and best_name is not None

    # Persist best model for API / Docker
    model_path = paths["models"] / "heart_disease_pipeline.joblib"
    joblib.dump(best_pipeline, model_path)

    meta = {
        "best_model": best_name,
        "best_roc_auc": best_auc,
        "best_params": {k: (None if v is None else v) for k, v in best_params.items()},
        "feature_order": list(x_train.columns),
        "model_path": str(model_path.relative_to(ROOT)),
    }
    meta_path = paths["models"] / "model_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    board_df = pd.DataFrame(leaderboard)
    board_path = paths["artifacts"] / "model_leaderboard.csv"
    board_df.to_csv(board_path, index=False)

    # Comparison bar chart
    plt.figure(figsize=(8, 4))
    sns.barplot(data=board_df, x="model", y="roc_auc")
    plt.title("Model Comparison (Test ROC-AUC)")
    plt.ylim(0.5, 1.0)
    plt.tight_layout()
    cmp_path = paths["artifacts"] / "model_comparison_roc_auc.png"
    plt.savefig(cmp_path, dpi=150)
    plt.close()

    with mlflow.start_run(run_name="best_model_selection"):
        mlflow.log_param("selected_model", best_name)
        mlflow.log_metric("selected_roc_auc", best_auc)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(meta_path))
        mlflow.log_artifact(str(board_path))
        mlflow.log_artifact(str(cmp_path))

    print("\nSelected best model:", best_name, "ROC-AUC=", round(best_auc, 4))
    print("Saved pipeline ->", model_path)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Train heart disease models")
    parser.add_argument("--experiment-name", default="heart-disease-uci")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()
    train_and_evaluate(
        experiment_name=args.experiment_name,
        test_size=args.test_size,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
    )


if __name__ == "__main__":
    main()
