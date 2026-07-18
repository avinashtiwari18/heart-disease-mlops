# Screenshots (for report submission)

## Demo video (generated)

| File | Description |
|------|-------------|
| `pipeline_demo.mp4` | ~60s walkthrough of the full MLOps pipeline |
| `pipeline_demo.gif` | Lightweight preview of the same demo |

Regenerate anytime:

```bash
source .venv/bin/activate
python scripts/record_demo.py
```

Capture and place additional images here before final PDF/DOCX packaging:

| File name (suggested) | What to capture |
|----------------------|-----------------|
| `01_eda_heatmap.png` | Correlation heatmap from notebook or `artifacts/eda/` |
| `02_mlflow_ui.png` | MLflow UI showing experiment runs |
| `03_model_comparison.png` | `artifacts/metrics/model_comparison_roc_auc.png` |
| `04_github_actions.png` | Green CI workflow run on GitHub Actions |
| `05_docker_predict.png` | Terminal showing `docker build/run` + curl `/predict` |
| `06_k8s_pods.png` | `kubectl get pods,svc` after Minikube deploy |
| `07_prometheus.png` | Prometheus targets / Grafana dashboard |
| `08_swagger_ui.png` | FastAPI `/docs` Swagger page |

Many plot artifacts are already generated under `../artifacts/` after you run EDA and training.
