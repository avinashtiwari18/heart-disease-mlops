# Heart Disease Risk Prediction – MLOps End-to-End Pipeline

**Course:** AIMLCZG523 – Machine Learning Operations  
**Assignment:** 01 (2026)  
**Problem:** Binary classification of heart-disease risk using the UCI Heart Disease dataset, with experiment tracking, CI/CD, Docker, Kubernetes, and monitoring.

---

## Repository layout

```
heart-disease-mlops/
├── api/                    # FastAPI serving (/predict, /health, /metrics)
├── src/                    # Data processing, features, EDA, training
├── scripts/                # Download data, sample predict, Minikube deploy
├── notebooks/              # EDA + training Jupyter notebooks
├── tests/                  # Pytest unit/API tests
├── models/                 # Serialized sklearn Pipeline + metadata
├── data/                   # raw / cleaned / processed CSVs
├── artifacts/              # EDA plots, metrics, API logs
├── mlflow_tracking/        # Local MLflow backend store
├── k8s/                    # Deployment, Service, Ingress, Helm, Prometheus
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── REPORT.md               # Written report (export to PDF/DOCX for submission)
└── screenshots/            # Place CI/CD & deployment screenshots here
```

---

## Quick start (clean machine)

```bash
cd AIML/heart-disease-mlops

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Data
python scripts/download_data.py

# 2) EDA plots -> artifacts/eda/
python -m src.eda

# 3) Train + MLflow -> models/ + mlflow_tracking/
python -m src.train

# 4) Tests
pytest tests -v

# 5) Run API locally
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Open http://127.0.0.1:8000/docs
bash scripts/sample_predict.sh
```

### MLflow UI

```bash
mlflow ui --backend-store-uri ./mlflow_tracking --port 5000
```

---

## Docker

```bash
# Requires trained model in models/
docker build -t heart-disease-api:1.0.0 .
docker run --rm -p 8000:8000 heart-disease-api:1.0.0

# Or compose
docker compose up --build -d
bash scripts/sample_predict.sh http://127.0.0.1:8000

# Optional Prometheus + Grafana
docker compose --profile monitoring up -d
# Prometheus: http://127.0.0.1:9090  Grafana: http://127.0.0.1:3000 (admin/admin)
```

### Sample `/predict` payload

```json
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
  "thal": 1
}
```

Response includes `prediction`, `prediction_label`, `confidence`, and `probability_disease`.

---

## Kubernetes (Minikube / Docker Desktop)

```bash
bash scripts/deploy_minikube.sh
kubectl port-forward svc/heart-disease-api-svc 8000:80
bash scripts/sample_predict.sh http://127.0.0.1:8000
```

Helm alternative:

```bash
eval "$(minikube docker-env)"
docker build -t heart-disease-api:1.0.0 .
helm upgrade --install hd-api k8s/helm/heart-disease-api
```

---

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on push/PR:

1. Install dependencies  
2. Lint (`flake8`)  
3. Download data  
4. Unit tests  
5. Train models (MLflow)  
6. API tests  
7. Upload artifacts  
8. Docker build + `/predict` smoke test  

Push this folder to a GitHub repository to activate the workflow. Pipeline **fails** if lint critical errors or tests fail.

---

## Monitoring & logging

- Structured request logs → stdout + `artifacts/logs/api_requests.log`
- Prometheus metrics → `GET /metrics` (via `prometheus-fastapi-instrumentator`)
- Compose/K8s Prometheus configs under `k8s/monitoring/`

---

## Dataset

- **Source:** [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) (Cleveland processed)
- **Script:** `python scripts/download_data.py`
- **Target:** binary `target` (0 = no disease, 1 = disease; original `num > 0`)

---

## Submission checklist

- [x] Code, Dockerfile, requirements  
- [x] Dataset download script + cleaned CSV (after download)  
- [x] Notebooks + training scripts  
- [x] `tests/` with Pytest  
- [x] GitHub Actions YAML  
- [x] K8s manifests + Helm chart  
- [x] Monitoring (Prometheus metrics + logging)  
- [ ] Screenshots in `screenshots/` (capture after you run CI/deploy)  
- [ ] Export `REPORT.md` → PDF/DOCX (~10 pages)  
- [ ] Short demo video of the pipeline  
- [ ] Public GitHub repo link in the report  

---

## License / academic integrity

Complete and personalize this project for your own submission. Do not copy another student’s pipelines, notebooks, or report text.
