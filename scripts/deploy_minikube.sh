#!/usr/bin/env bash
# Build image inside Minikube and deploy Kubernetes manifests.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Ensure Minikube is running"
minikube status >/dev/null 2>&1 || minikube start

echo "==> Use Minikube Docker daemon"
eval "$(minikube docker-env)"

echo "==> Build API image"
docker build -t heart-disease-api:1.0.0 .

echo "==> Apply manifests"
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

echo "==> Wait for rollout"
kubectl rollout status deployment/heart-disease-api --timeout=180s

echo "==> Service endpoints"
kubectl get svc heart-disease-api-svc heart-disease-api-nodeport

echo
echo "Access options:"
echo "  1) minikube service heart-disease-api-nodeport --url"
echo "  2) kubectl port-forward svc/heart-disease-api-svc 8000:80"
echo "Then: bash scripts/sample_predict.sh http://127.0.0.1:8000"
