# Screenshots (for report submission)

## Demo video + operational screenshots

| File | Description |
|------|-------------|
| `pipeline_demo.mp4` | ~60s walkthrough of the full MLOps pipeline |
| `02_mlflow_ui.png` | MLflow UI — heart-disease-uci experiment runs |
| `04_github_actions.png` | Green GitHub Actions CI run |
| `05_docker_predict.png` | Docker build/run + `/health` + `/predict` |
| `06_k8s_pods.png` | Kubernetes pods/services (Rancher Desktop) |
| `06_k8s_predict.png` | K8s port-forward prediction smoke test |
| `07_prometheus_or_metrics.png` | API `/metrics` Prometheus exposition |
| `07_prometheus_targets.png` | Prometheus targets page |
| `08_swagger_ui.png` | FastAPI Swagger `/docs` |
| `class_balance.png` / `correlation_heatmap.png` / … | EDA plots |

Regenerate demo video:

```bash
source .venv/bin/activate
python scripts/record_demo.py
```

Re-capture operational screenshots (requires Docker/Rancher running):

```bash
source .venv/bin/activate
python scripts/capture_operational_screenshots.py
```
