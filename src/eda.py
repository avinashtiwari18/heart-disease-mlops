#!/usr/bin/env python3
"""Generate professional EDA visualizations for the Heart Disease UCI dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_processing import (  # noqa: E402
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    clean_dataframe,
    load_raw_dataframe,
    missing_value_report,
)


def run_eda() -> Path:
    out_dir = ROOT / "artifacts" / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_raw_dataframe()
    missing = missing_value_report(df_raw)
    missing.to_csv(out_dir / "missing_value_report.csv")

    df = clean_dataframe(df_raw)
    sns.set_theme(style="whitegrid", context="notebook")

    # 1) Class balance
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(
        data=df,
        x=TARGET_COLUMN,
        hue=TARGET_COLUMN,
        palette=["#2a9d8f", "#e76f51"],
        legend=False,
    )
    ax.set_title("Class Balance: Heart Disease Presence")
    ax.set_xlabel("Target (0 = No Disease, 1 = Disease)")
    ax.set_ylabel("Count")
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(out_dir / "class_balance.png", dpi=150)
    plt.close()

    # 2) Histograms of numeric features
    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color="#264653")
        axes[i].set_title(f"Distribution: {col}")
    axes[-1].axis("off")
    fig.suptitle("Numeric Feature Histograms", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "numeric_histograms.png", dpi=150)
    plt.close()

    # 3) Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df[FEATURE_COLUMNS + [TARGET_COLUMN]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_heatmap.png", dpi=150)
    plt.close()

    # 4) Feature relationships vs target
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sns.boxplot(
        data=df, x=TARGET_COLUMN, y="age", hue=TARGET_COLUMN, ax=axes[0], palette="Set2", legend=False
    )
    axes[0].set_title("Age vs Target")
    sns.boxplot(
        data=df, x=TARGET_COLUMN, y="thalach", hue=TARGET_COLUMN, ax=axes[1], palette="Set2", legend=False
    )
    axes[1].set_title("Max Heart Rate vs Target")
    sns.boxplot(
        data=df, x=TARGET_COLUMN, y="oldpeak", hue=TARGET_COLUMN, ax=axes[2], palette="Set2", legend=False
    )
    axes[2].set_title("ST Depression vs Target")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_relationships.png", dpi=150)
    plt.close()

    # 5) Missing values bar chart (raw)
    plt.figure(figsize=(8, 4))
    missing["missing_count"].plot(kind="bar", color="#e9c46a")
    plt.title("Missing Value Counts (Raw Data)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "missing_values.png", dpi=150)
    plt.close()

    # Summary stats
    summary = df.describe(include="all").T
    summary.to_csv(out_dir / "summary_statistics.csv")

    # Target correlation ranking
    target_corr = (
        corr[TARGET_COLUMN]
        .drop(TARGET_COLUMN)
        .abs()
        .sort_values(ascending=False)
        .reset_index()
    )
    target_corr.columns = ["feature", "abs_corr_with_target"]
    target_corr.to_csv(out_dir / "target_correlation_ranking.csv", index=False)

    print(f"EDA artifacts written to {out_dir}")
    return out_dir


if __name__ == "__main__":
    run_eda()
