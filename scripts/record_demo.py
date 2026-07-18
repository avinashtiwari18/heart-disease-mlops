#!/usr/bin/env python3
"""
Generate a short demo video of the overall MLOps pipeline (assignment deliverable).

Covers (~3 minutes):
  repo → EDA → training/MLflow → CI/CD → Docker/API → Kubernetes → monitoring

Output: screenshots/pipeline_demo.mp4
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
    draw.rounded_rectangle((80, 140, W - 80, H - 140), radius=24, fill=CARD)
    draw.text((110, 180), title, font=font(40, bold=True), fill=TEXT)
    y = draw_wrapped(draw, subtitle, (110, 260), font(24), fill=MUTED, width=72)
    if footer:
        draw.text((110, max(y + 28, 500)), footer, font=font(20), fill=ACCENT)
    return img


def bullets_slide(title: str, bullets: list[str]) -> Image.Image:
    img, draw = new_slide()
    draw.text((70, 40), title, font=font(34, bold=True), fill=TEXT)
    draw.line((70, 90, W - 70, 90), fill=ACCENT, width=3)
    y = 120
    for b in bullets:
        draw.ellipse((78, y + 10, 94, y + 26), fill=ACCENT)
        y = draw_wrapped(draw, b, (120, y), font(23), fill=TEXT, width=74)
        y += 14
    return img


def image_slide(title: str, image_path: Path, caption: str = "") -> Image.Image:
    img, draw = new_slide()
    draw.text((60, 24), title, font=font(28, bold=True), fill=TEXT)
    panel = Image.open(image_path).convert("RGB")
    max_w, max_h = W - 120, H - 140
    panel.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    x = (W - panel.width) // 2
    y = 78
    draw.rounded_rectangle(
        (x - 8, y - 8, x + panel.width + 8, y + panel.height + 8),
        radius=10,
        fill=CARD,
    )
    img.paste(panel, (x, y))
    if caption:
        draw.text((60, H - 48), caption, font=font(18), fill=MUTED)
    return img


def code_slide(title: str, lines: list[str]) -> Image.Image:
    img, draw = new_slide()
    draw.text((60, 30), title, font=font(28, bold=True), fill=TEXT)
    draw.rounded_rectangle((50, 85, W - 50, H - 50), radius=16, fill=(12, 18, 28))
    y = 110
    mono = font(18)
    for path in (
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
    ):
        try:
            mono = ImageFont.truetype(path, size=18)
            break
        except OSError:
            pass
    for line in lines:
        color = ACCENT if line.startswith("$") or line.startswith(">") else TEXT
        if line.startswith("#"):
            color = MUTED
        draw.text((80, y), line[:95], font=mono, fill=color)
        y += 32
    return img


def to_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"))


def hold(frames: list[np.ndarray], img: Image.Image, seconds: float) -> None:
    arr = to_array(img)
    for _ in range(max(1, int(seconds * FPS))):
        frames.append(arr)


def maybe_image(
    frames: list[np.ndarray],
    title: str,
    path: Path,
    caption: str,
    seconds: float,
) -> None:
    if path.exists():
        hold(frames, image_slide(title, path, caption), seconds)
    else:
        hold(frames, bullets_slide(title, [f"Missing artifact: {path.name}", caption]), 2.0)


def build_frames() -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    shots = OUT_DIR
    eda = ROOT / "artifacts" / "eda"
    metrics = ROOT / "artifacts" / "metrics"

    # 1) Title / intro
    hold(
        frames,
        title_slide(
            "Heart Disease MLOps Pipeline Demo",
            "AIMLCZG523 Assignment 01 — AVINASH TIWARI [2024AC05513]. "
            "End-to-end: data → EDA → training → MLflow → CI/CD → Docker → Kubernetes → monitoring.",
            "Repo: github.com/avinashtiwari18/heart-disease-mlops",
        ),
        6.0,
    )

    # 2) Repo walkthrough
    hold(
        frames,
        bullets_slide(
            "1) Repository Walkthrough",
            [
                "Public GitHub: avinashtiwari18/heart-disease-mlops",
                "src/ — data processing, EDA, feature pipeline, training",
                "api/ — FastAPI /predict, /health, /metrics",
                "tests/ + .github/workflows/ci.yml — automated quality gates",
                "Dockerfile + k8s/ — containerization and Kubernetes deploy",
                "REPORT.pdf — written submission with evidence screenshots",
            ],
        ),
        7.0,
    )

    hold(
        frames,
        code_slide(
            "1) Clean setup commands",
            [
                "$ git clone https://github.com/avinashtiwari18/heart-disease-mlops.git",
                "$ cd heart-disease-mlops && python3 -m venv .venv && source .venv/bin/activate",
                "$ pip install -r requirements.txt",
                "$ python scripts/download_data.py",
                "$ python -m src.eda && python -m src.train",
                "$ pytest tests -v",
                "$ uvicorn api.main:app --port 8000",
            ],
        ),
        7.0,
    )

    # 3) Data + EDA
    hold(
        frames,
        code_slide(
            "2) Data Acquisition (UCI Heart Disease)",
            [
                "$ python scripts/download_data.py",
                "# Source: UCI processed.cleveland.data",
                "# Output: data/cleaned/heart_disease_cleaned.csv",
                "# Shape: 303 rows × 14 columns",
                "# Target balance: 0=164 (no disease), 1=139 (disease)",
            ],
        ),
        6.0,
    )

    for title, path, caption, secs in [
        (
            "2) EDA — Class Balance",
            shots / "class_balance.png" if (shots / "class_balance.png").exists() else eda / "class_balance.png",
            "Near-balanced binary target",
            5.0,
        ),
        (
            "2) EDA — Correlation Heatmap",
            shots / "correlation_heatmap.png" if (shots / "correlation_heatmap.png").exists() else eda / "correlation_heatmap.png",
            "Feature relationships with heart-disease target",
            5.5,
        ),
        (
            "2) EDA — Numeric Distributions",
            shots / "numeric_histograms.png" if (shots / "numeric_histograms.png").exists() else eda / "numeric_histograms.png",
            "Clinical feature histograms",
            5.0,
        ),
    ]:
        maybe_image(frames, title, path, caption, secs)

    # 4) Training + MLflow
    hold(
        frames,
        code_slide(
            "3) Model Training + Experiment Tracking",
            [
                "$ python -m src.train",
                "# Models: LogisticRegression | RandomForest | GradientBoosting",
                "# Tuning: GridSearchCV + 5-fold Stratified CV (ROC-AUC)",
                "# Best model: logistic_regression  test ROC-AUC = 0.965",
                "# Saved: models/heart_disease_pipeline.joblib",
                "$ mlflow ui --backend-store-uri ./mlflow_tracking --port 5000",
            ],
        ),
        7.5,
    )

    maybe_image(
        frames,
        "3) Model Comparison (Test ROC-AUC)",
        shots / "model_comparison_roc_auc.png"
        if (shots / "model_comparison_roc_auc.png").exists()
        else metrics / "model_comparison_roc_auc.png",
        "Selected: logistic_regression (highest ROC-AUC)",
        5.5,
    )
    maybe_image(
        frames,
        "3) MLflow UI — Experiment Runs",
        shots / "02_mlflow_ui.png",
        "Logged params, metrics, sklearn models for each run",
        8.0,
    )
    maybe_image(
        frames,
        "3) Best Model ROC Curve",
        metrics / "logistic_regression_roc_curve.png",
        "Held-out test discrimination for selected pipeline",
        5.0,
    )

    # 5) CI/CD
    hold(
        frames,
        code_slide(
            "4) CI/CD — GitHub Actions",
            [
                "# Workflow: .github/workflows/ci.yml",
                "$ flake8 src api tests scripts",
                "$ python scripts/download_data.py",
                "$ pytest tests -v",
                "$ python -m src.train",
                "$ docker build -t heart-disease-api:ci .",
                "$ curl /health && curl /predict   # container smoke test",
                "# Pipeline fails on lint/test errors",
            ],
        ),
        7.0,
    )
    maybe_image(
        frames,
        "4) GitHub Actions — Green CI Run",
        shots / "04_github_actions.png",
        "Heart Disease MLOps CI #1 on main (~3m 25s)",
        8.0,
    )

    # 6) Docker + API
    hold(
        frames,
        code_slide(
            "5) Dockerized FastAPI Serving",
            [
                "$ docker build -t heart-disease-api:1.0.0 .",
                "$ docker run -d --name hd-api -p 8000:8000 heart-disease-api:1.0.0",
                "$ curl http://127.0.0.1:8000/health",
                '> {"status":"ok","model_loaded":true,"model_name":"logistic_regression"}',
                "$ curl -X POST http://127.0.0.1:8000/predict -d '{patient JSON}'",
                '> {"prediction":0,"prediction_label":"low_risk","confidence":0.8148}',
                "# Swagger UI: http://127.0.0.1:8000/docs",
            ],
        ),
        7.5,
    )
    maybe_image(
        frames,
        "5) Docker — /health + /predict",
        shots / "05_docker_predict.png",
        "Isolated container serves trained sklearn Pipeline",
        8.0,
    )
    maybe_image(
        frames,
        "5) FastAPI Swagger UI (/docs)",
        shots / "08_swagger_ui.png",
        "Interactive API contract for /predict and /metrics",
        5.5,
    )

    # 7) Kubernetes
    hold(
        frames,
        code_slide(
            "6) Kubernetes Deployment (Rancher Desktop)",
            [
                "$ docker build -t heart-disease-api:1.0.0 .",
                "$ kubectl apply -f k8s/deployment.yaml",
                "$ kubectl apply -f k8s/service.yaml",
                "$ kubectl rollout status deployment/heart-disease-api",
                "$ kubectl get pods,svc -o wide",
                "# 2/2 replicas Running",
                "$ kubectl port-forward svc/heart-disease-api-svc 8000:80",
            ],
        ),
        7.0,
    )
    maybe_image(
        frames,
        "6) Kubernetes — Pods & Services",
        shots / "06_k8s_pods.png",
        "Deployment + LoadBalancer/NodePort services",
        8.0,
    )
    maybe_image(
        frames,
        "6) Kubernetes — Port-forward Inference",
        shots / "06_k8s_predict.png",
        "Verified /health and /predict via cluster service",
        5.5,
    )

    # 8) Monitoring
    hold(
        frames,
        bullets_slide(
            "7) Monitoring & Logging",
            [
                "Request logging middleware → stdout + api_requests.log",
                "Prometheus metrics exposed at GET /metrics",
                "docker compose --profile monitoring → Prometheus + Grafana",
                "Useful for downtime, latency, and 5xx detection",
            ],
        ),
        6.0,
    )
    maybe_image(
        frames,
        "7) Prometheus /metrics Endpoint",
        shots / "07_prometheus_or_metrics.png",
        "HTTP request metrics from instrumented FastAPI app",
        5.5,
    )
    maybe_image(
        frames,
        "7) Prometheus Targets UI",
        shots / "07_prometheus_targets.png",
        "Scrape configuration for heart-disease-api",
        7.0,
    )

    # Closing
    hold(
        frames,
        title_slide(
            "Pipeline Demo Complete",
            "Assignment deliverables covered: reproducible training, MLflow tracking, "
            "CI/CD, Dockerized API, Kubernetes deploy, monitoring, report, and this video.",
            "Thank you — AIMLCZG523 Assignment 01",
        ),
        6.0,
    )
    return frames


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Building assignment demo frames...")
    frames = build_frames()
    mp4_path = OUT_DIR / "pipeline_demo.mp4"
    gif_path = OUT_DIR / "pipeline_demo.gif"

    duration = len(frames) / FPS
    print(f"Writing {mp4_path} ({len(frames)} frames @ {FPS} fps, ~{duration:.0f}s)...")
    imageio.mimsave(
        mp4_path,
        frames,
        fps=FPS,
        codec="libx264",
        quality=7,
        pixelformat="yuv420p",
        macro_block_size=1,
    )

    # Shorter GIF preview
    preview = frames[::3]
    print(f"Writing preview GIF {gif_path}...")
    imageio.mimsave(gif_path, preview, fps=FPS, loop=0)

    print(f"Done. Duration ≈ {duration:.1f}s (~{duration/60:.1f} min)")
    print(f"  MP4: {mp4_path}")
    print(f"  GIF: {gif_path}")


if __name__ == "__main__":
    main()
