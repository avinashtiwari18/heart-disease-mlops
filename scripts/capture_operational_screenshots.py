#!/usr/bin/env python3
"""Capture operational screenshots for assignment report."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "screenshots"
W, H = 1400, 900


def font(size: int, bold: bool = False):
    paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def terminal_shot(path: Path, title: str, lines: list[str]) -> None:
    img = Image.new("RGB", (W, H), (18, 24, 32))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, W, 48), fill=(32, 42, 56))
    draw.text((20, 12), title, font=font(22, True), fill=(230, 237, 243))
    y = 70
    mono = font(18)
    for line in lines:
        color = (126, 231, 135) if line.startswith("$") else (230, 237, 243)
        if line.startswith("#") or line.startswith("==="):
            color = (121, 192, 255)
        draw.text((24, y), line[:140], font=mono, fill=color)
        y += 26
        if y > H - 30:
            break
    img.save(path)
    print(f"Saved {path}")


def screenshot_url(url: str, path: Path, wait_ms: int = 2500, full: bool = True) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto(url, wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(wait_ms)
        page.screenshot(path=str(path), full_page=full)
        browser.close()
    print(f"Saved {path}")


def run(cmd: list[str] | str, cwd: Path | None = None, timeout: int = 600) -> str:
    print("+", cmd if isinstance(cmd, str) else " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=cwd or ROOT,
        shell=isinstance(cmd, str),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        print(out[-2000:])
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd}")
    return out


def main() -> None:
    SHOTS.mkdir(parents=True, exist_ok=True)
    env_path = str(ROOT / ".venv" / "bin")

    # ---- MLflow UI ----
    mlflow_bin = ROOT / ".venv" / "bin" / "mlflow"
    mlflow_proc = subprocess.Popen(
        [
            str(mlflow_bin),
            "ui",
            "--backend-store-uri",
            str(ROOT / "mlflow_tracking"),
            "--host",
            "127.0.0.1",
            "--port",
            "5000",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(4)
        screenshot_url(
            "http://127.0.0.1:5000/#/experiments/1",
            SHOTS / "02_mlflow_ui.png",
            wait_ms=3500,
        )
        # Fallback experiment list if #/experiments/1 is blank
        screenshot_url(
            "http://127.0.0.1:5000/",
            SHOTS / "02_mlflow_ui_home.png",
            wait_ms=2500,
        )
    finally:
        mlflow_proc.terminate()
        try:
            mlflow_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            mlflow_proc.kill()

    # ---- Docker build + run ----
    run(["docker", "build", "-t", "heart-disease-api:1.0.0", "."], timeout=900)
    subprocess.run(["docker", "rm", "-f", "hd-api"], capture_output=True)
    run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            "hd-api",
            "-p",
            "8000:8000",
            "heart-disease-api:1.0.0",
        ]
    )
    time.sleep(8)
    health = run(["curl", "-s", "http://127.0.0.1:8000/health"])
    predict = run(
        [
            "curl",
            "-s",
            "-X",
            "POST",
            "http://127.0.0.1:8000/predict",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps(
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
            ),
        ]
    )
    terminal_shot(
        SHOTS / "05_docker_predict.png",
        "Docker: heart-disease-api /health + /predict",
        [
            "$ docker build -t heart-disease-api:1.0.0 .",
            "# Build succeeded",
            "$ docker run -d --name hd-api -p 8000:8000 heart-disease-api:1.0.0",
            "$ curl -s http://127.0.0.1:8000/health | python -m json.tool",
            *json.dumps(json.loads(health), indent=2).splitlines(),
            "$ curl -s -X POST http://127.0.0.1:8000/predict ...",
            *json.dumps(json.loads(predict), indent=2).splitlines(),
        ],
    )
    screenshot_url("http://127.0.0.1:8000/docs", SHOTS / "08_swagger_ui.png", wait_ms=2000)

    # ---- Metrics ----
    metrics = run(["curl", "-s", "http://127.0.0.1:8000/metrics"])
    metric_lines = [ln for ln in metrics.splitlines() if ln and not ln.startswith("#")][:25]
    comment_lines = [ln for ln in metrics.splitlines() if ln.startswith("# HELP") or ln.startswith("# TYPE")][:10]
    terminal_shot(
        SHOTS / "07_prometheus_or_metrics.png",
        "Prometheus metrics endpoint: GET /metrics",
        [
            "$ curl -s http://127.0.0.1:8000/metrics | head",
            *comment_lines,
            *metric_lines,
        ],
    )

    # Optional Prometheus UI via compose profile (reuse built image; free port 8000 first)
    try:
        subprocess.run(["docker", "rm", "-f", "hd-api"], capture_output=True)
        run(["docker", "compose", "--profile", "monitoring", "up", "-d", "--build"], timeout=600)
        time.sleep(12)
        screenshot_url(
            "http://127.0.0.1:9090/targets",
            SHOTS / "07_prometheus_targets.png",
            wait_ms=3500,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Prometheus compose optional step skipped/failed: {exc}")
        # Keep a standalone API container for later K8s-independent checks
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                "hd-api",
                "-p",
                "8000:8000",
                "heart-disease-api:1.0.0",
            ],
            capture_output=True,
        )

    # ---- Kubernetes ----
    # Ensure image exists for cluster (same docker daemon on Rancher)
    run(["kubectl", "apply", "-f", "k8s/deployment.yaml"])
    run(["kubectl", "apply", "-f", "k8s/service.yaml"])
    try:
        run(
            [
                "kubectl",
                "rollout",
                "status",
                "deployment/heart-disease-api",
                "--timeout=180s",
            ],
            timeout=200,
        )
    except RuntimeError:
        # print diagnostics but continue to capture whatever state exists
        print(run(["kubectl", "get", "pods,svc", "-o", "wide"]))
        print(run(["kubectl", "describe", "deployment/heart-disease-api"])[-1500:])

    pods = run(["kubectl", "get", "pods,svc", "-o", "wide"])
    terminal_shot(
        SHOTS / "06_k8s_pods.png",
        "Kubernetes: heart-disease-api deployment",
        [
            "$ kubectl apply -f k8s/deployment.yaml",
            "$ kubectl apply -f k8s/service.yaml",
            "$ kubectl rollout status deployment/heart-disease-api",
            "$ kubectl get pods,svc -o wide",
            *pods.splitlines(),
        ],
    )

    # Port-forward smoke if pods ready
    if "Running" in pods:
        pf = subprocess.Popen(
            [
                "kubectl",
                "port-forward",
                "svc/heart-disease-api-svc",
                "8001:80",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(3)
            k8s_health = run(["curl", "-s", "http://127.0.0.1:8001/health"])
            terminal_shot(
                SHOTS / "06_k8s_predict.png",
                "Kubernetes port-forward /health",
                [
                    "$ kubectl port-forward svc/heart-disease-api-svc 8001:80",
                    "$ curl -s http://127.0.0.1:8001/health",
                    *json.dumps(json.loads(k8s_health), indent=2).splitlines(),
                ],
            )
        finally:
            pf.terminate()

    print("\nAll operational screenshots written to", SHOTS)


if __name__ == "__main__":
    main()
