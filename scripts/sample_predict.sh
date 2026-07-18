#!/usr/bin/env bash
# Smoke-test the local or remote /predict endpoint.
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8000}"

echo "Health check: ${BASE_URL}/health"
curl -fsS "${BASE_URL}/health" | python -m json.tool

echo
echo "Prediction request:"
curl -fsS -X POST "${BASE_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }' | python -m json.tool
