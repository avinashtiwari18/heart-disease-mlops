#!/usr/bin/env python3
"""
Generate a short demo video of the Heart Disease MLOps pipeline.

Output: screenshots/pipeline_demo.mp4 (and .gif preview)
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "screenshots"
W, H = 1280, 720
BG = (18, 28, 40)
CARD = (28, 42, 58)
ACCENT = (42, 157, 143)
TEXT = (240, 244, 248)
MUTED = (170, 184, 198)
FPS = 2


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def new_slide() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, W, 8), fill=ACCENT)
    draw.rectangle((0, H - 8, W, H), fill=ACCENT)
    return img, draw


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    font_obj,
    fill=TEXT,
    width: int = 70,
    line_gap: int = 8,
) -> int:
    x, y = xy
    for line in textwrap.wrap(text, width=width):
        draw.text((x, y), line, font=font_obj, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font_obj)
        y = bbox[3] + line_gap
    return y


def title_slide(title: str, subtitle: str, footer: str = "") -> Image.Image:
    img, draw = new_slide()
    draw.rounded_rectangle((80, 160, W - 80, H - 160), radius=24, fill=CARD)
    draw.text((110, 210), title, font=font(44, bold=True), fill=TEXT)
    y = draw_wrapped(draw, subtitle, (110, 290), font(26), fill=MUTED, width=70)
    if footer:
        draw.text((110, max(y + 30, 520)), footer, font=font(20), fill=ACCENT)
    return img


def bullets_slide(title: str, bullets: list[str]) -> Image.Image:
    img, draw = new_slide()
    draw.text((70, 50), title, font=font(36, bold=True), fill=TEXT)
    draw.line((70, 100, W - 70, 100), fill=ACCENT, width=3)
    y = 140
    for b in bullets:
        draw.ellipse((78, y + 10, 94, y + 26), fill=ACCENT)
        y = draw_wrapped(draw, b, (120, y), font(24), fill=TEXT, width=72)
        y += 18
    return img


def image_slide(title: str, image_path: Path, caption: str = "") -> Image.Image:
    img, draw = new_slide()
    draw.text((70, 30), title, font=font(30, bold=True), fill=TEXT)
    panel = Image.open(image_path).convert("RGB")
    max_w, max_h = W - 140, H - 160
    panel.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    x = (W - panel.width) // 2
    y = 90
    draw.rounded_rectangle(
        (x - 10, y - 10, x + panel.width + 10, y + panel.height + 10),
        radius=12,
        fill=CARD,
    )
    img.paste(panel, (x, y))
    if caption:
        draw.text((70, H - 55), caption, font=font(18), fill=MUTED)
    return img


def code_slide(title: str, lines: list[str]) -> Image.Image:
    img, draw = new_slide()
    draw.text((70, 40), title, font=font(30, bold=True), fill=TEXT)
    draw.rounded_rectangle((60, 100, W - 60, H - 60), radius=16, fill=(12, 18, 28))
    y = 130
    mono_candidates = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
    ]
    mono = font(20)
    for path in mono_candidates:
        try:
            mono = ImageFont.truetype(path, size=20)
            break
        except OSError:
            pass
    for line in lines:
        color = ACCENT if line.startswith("$") or line.startswith(">") else TEXT
        if line.startswith("#"):
            color = MUTED
        draw.text((90, y), line, font=mono, fill=color)
        y += 34
    return img


def to_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"))


def hold(frames: list[np.ndarray], img: Image.Image, seconds: float) -> None:
    arr = to_array(img)
    for _ in range(max(1, int(seconds * FPS))):
        frames.append(arr)


def build_frames() -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    eda = ROOT / "artifacts" / "eda"
    metrics = ROOT / "artifacts" / "metrics"

    hold(
        frames,
        title_slide(
            "Heart Disease MLOps Pipeline",
            "AIMLCZG523 Assignment 01 — End-to-end ML development, CI/CD, Docker, Kubernetes & monitoring",
            "Demo walkthrough (~2 minutes)",
        ),
        3.5,
    )

    hold(
        frames,
        bullets_slide(
            "Pipeline Stages",
            [
                "1. Download & clean UCI Heart Disease dataset",
                "2. EDA — histograms, correlation, class balance",
                "3. Train models with GridSearchCV + MLflow tracking",
                "4. Package sklearn Pipeline (joblib) for inference",
                "5. Serve via FastAPI /predict + Prometheus /metrics",
                "6. Containerize (Docker) and deploy (Kubernetes)",
                "7. Automate with GitHub Actions CI/CD + Pytest",
            ],
        ),
        5.0,
    )

    hold(
        frames,
        code_slide(
            "1) Data Acquisition",
            [
                "$ python scripts/download_data.py",
                "# Source: UCI processed.cleveland.data",
                "# Saved: data/cleaned/heart_disease_cleaned.csv",
                "# Shape: 303 rows × 14 columns",
                "# Target: 0=164 (no disease), 1=139 (disease)",
            ],
        ),
        4.0,
    )

    for title, path, caption, secs in [
        ("2) EDA — Class Balance", eda / "class_balance.png", "Near-balanced binary target", 3.0),
        ("2) EDA — Correlation Heatmap", eda / "correlation_heatmap.png", "Feature relationships with target", 3.5),
        ("2) EDA — Numeric Distributions", eda / "numeric_histograms.png", "Clinical feature spreads", 3.0),
        ("2) EDA — Feature Relationships", eda / "feature_relationships.png", "Age / thalach / oldpeak vs target", 3.0),
    ]:
        if path.exists():
            hold(frames, image_slide(title, path, caption), secs)

    hold(
        frames,
        code_slide(
            "3) Model Training + MLflow",
            [
                "$ python -m src.eda",
                "$ python -m src.train",
                "# Models: LogisticRegression | RandomForest | GradientBoosting",
                "# Selection metric: test ROC-AUC (5-fold CV + GridSearch)",
                "# Best: logistic_regression  ROC-AUC = 0.965",
                "# Artifact: models/heart_disease_pipeline.joblib",
            ],
        ),
        5.0,
    )

    if (metrics / "model_comparison_roc_auc.png").exists():
        hold(
            frames,
            image_slide(
                "3) Model Comparison (Test ROC-AUC)",
                metrics / "model_comparison_roc_auc.png",
                "Winner: Logistic Regression (0.965)",
            ),
            3.5,
        )
    if (metrics / "logistic_regression_roc_curve.png").exists():
        hold(
            frames,
            image_slide(
                "3) Best Model ROC Curve",
                metrics / "logistic_regression_roc_curve.png",
                "Strong discrimination on held-out test set",
            ),
            3.0,
        )
    if (metrics / "logistic_regression_confusion_matrix.png").exists():
        hold(
            frames,
            image_slide(
                "3) Confusion Matrix — Logistic Regression",
                metrics / "logistic_regression_confusion_matrix.png",
                "Accuracy 0.885 | Precision 0.839 | Recall 0.929",
            ),
            3.0,
        )

    hold(
        frames,
        code_slide(
            "4) API Inference (FastAPI)",
            [
                "$ uvicorn api.main:app --port 8000",
                "> GET /health",
                '  {"status":"ok","model_loaded":true,"model_name":"logistic_regression"}',
                "> POST /predict  (sample patient JSON)",
                '  {"prediction":0,"prediction_label":"low_risk",',
                '   "confidence":0.8148,"probability_disease":0.1852}',
                "# Also: /docs (Swagger)  |  /metrics (Prometheus)",
            ],
        ),
        5.5,
    )

    hold(
        frames,
        bullets_slide(
            "5) Production Packaging",
            [
                "Docker: Dockerfile builds isolated API + model image",
                "Compose: docker compose up --build  → localhost:8000",
                "Kubernetes: k8s/deployment.yaml + NodePort/LoadBalancer",
                "Helm chart: k8s/helm/heart-disease-api/",
                "Deploy helper: bash scripts/deploy_minikube.sh",
                "Monitoring: request logs + Prometheus scrape of /metrics",
            ],
        ),
        5.0,
    )

    hold(
        frames,
        code_slide(
            "6) CI/CD — GitHub Actions",
            [
                "# .github/workflows/ci.yml",
                "$ flake8 src api tests scripts",
                "$ python scripts/download_data.py",
                "$ pytest tests -v",
                "$ python -m src.train",
                "$ docker build -t heart-disease-api:ci .",
                "$ curl /health && curl /predict   # smoke test",
                "# Pipeline fails on lint/test errors (clear logs)",
            ],
        ),
        5.0,
    )

    hold(
        frames,
        title_slide(
            "Pipeline Demo Complete",
            "Code, model, Docker, K8s manifests, tests, MLflow tracking, and report are in AIML/heart-disease-mlops",
            "Thank you — AIMLCZG523 Assignment 01",
        ),
        4.0,
    )
    return frames


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Building demo frames...")
    frames = build_frames()
    mp4_path = OUT_DIR / "pipeline_demo.mp4"
    gif_path = OUT_DIR / "pipeline_demo.gif"

    print(f"Writing {mp4_path} ({len(frames)} frames @ {FPS} fps)...")
    imageio.mimsave(
        mp4_path,
        frames,
        fps=FPS,
        codec="libx264",
        quality=7,
        pixelformat="yuv420p",
        macro_block_size=1,
    )

    # Lightweight GIF preview (downsample temporally)
    preview = frames[::2]
    print(f"Writing preview GIF {gif_path}...")
    imageio.mimsave(gif_path, preview, fps=FPS, loop=0)

    duration = len(frames) / FPS
    print(f"Done. Duration ≈ {duration:.1f}s")
    print(f"  MP4: {mp4_path}")
    print(f"  GIF: {gif_path}")


if __name__ == "__main__":
    main()
