# Finance AI Engineer — Commands Reference

## Local Development

### Setup
```bash
cd ~/finance-ai-engineer
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### Run locally
```bash
python3 -m uvicorn api_service:app --reload --port 8000
```

### Test locally
```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Basel III capital ratio?"}'

curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"email": "Hi team, meeting Monday 2pm. All must attend."}'
```

### Run full system
```bash
python3 finance_ai_system.py
```

### Run CI/CD pipeline locally
```bash
python3 validate_data.py && \
python3 evaluate.py && \
python3 merge_lora.py
```

---

## Docker

### Build image (Mac M1/M2 — AMD64 for GKE)
```bash
docker buildx build \
  --platform linux/amd64 \
  -t gcr.io/mlops-lab-prassadh-2026/finance-ai-model:latest \
  --push \
  .
```

### Build image (local testing)
```bash
docker build -t finance-ai-engineer .
```

### Run container locally
```bash
docker run -p 8000:8000 --env-file .env finance-ai-engineer
```

### Run with docker-compose
```bash
docker-compose up
```

---

## GCP Setup

### Authenticate
```bash
gcloud auth login
gcloud config set project mlops-lab-prassadh-2026
```

### Enable APIs
```bash
gcloud services enable container.googleapis.com \
  containerregistry.googleapis.com \
  cloudbuild.googleapis.com
```

### Configure Docker for GCR
```bash
gcloud auth configure-docker
```

---

## GKE — Create Cluster

### Create cluster
```bash
gcloud container clusters create finance-ai-cluster \
  --zone us-central1-a \
  --num-nodes 2 \
  --machine-type e2-medium \
  --disk-size 20GB
```

### Connect kubectl
```bash
gcloud container clusters get-credentials finance-ai-cluster \
  --zone us-central1-a
```

### Verify nodes
```bash
kubectl get nodes
```

---

## Kubernetes — Deploy

### Create secrets
```bash
kubectl create secret generic finance-ai-secrets \
  --from-literal=anthropic-api-key=$(grep ANTHROPIC_API_KEY .env | cut -d '=' -f2) \
  --from-literal=langchain-api-key=$(grep LANGCHAIN_API_KEY .env | cut -d '=' -f2)
```

### Deploy all manifests
```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

### Scale to 1 replica (e2-medium nodes)
```bash
kubectl scale deployment/finance-ai-api --replicas=1
```

### Get external IP
```bash
kubectl get service finance-ai-service
```

---

## Kubernetes — Monitor

### Check pods
```bash
kubectl get pods
```

### Check all resources
```bash
kubectl get all
```

### Check logs
```bash
kubectl logs -f deployment/finance-ai-api
```

### Describe pod (debug)
```bash
kubectl describe pod <pod-name>
```

### Check HPA
```bash
kubectl get hpa
```

---

## Kubernetes — Update

### Restart deployment
```bash
kubectl rollout restart deployment/finance-ai-api
```

### Check rollout status
```bash
kubectl rollout status deployment/finance-ai-api
```

### Scale up
```bash
kubectl scale deployment/finance-ai-api --replicas=3
```

---

## Kubernetes — Cleanup

### Delete all resources
```bash
kubectl delete -f k8s/
kubectl delete secret finance-ai-secrets
```

### Delete cluster (SAVES MONEY)
```bash
gcloud container clusters delete finance-ai-cluster \
  --zone us-central1-a
```

---

## Test Live API

### Replace IP with your external IP
```bash
EXTERNAL_IP=34.9.87.94

# Health check
curl http://$EXTERNAL_IP/health

# RAG query
curl -X POST http://$EXTERNAL_IP/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Basel III capital ratio?"}'

# Email summarization
curl -X POST http://$EXTERNAL_IP/summarize \
  -H "Content-Type: application/json" \
  -d '{"email": "Hi team, meeting Monday 2pm. All must attend."}'

# Batch (concurrent)
curl -X POST http://$EXTERNAL_IP/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"question": "What is Basel III?"},
    {"question": "What are AML requirements?"},
    {"question": "What is the Volcker Rule?"}
  ]'
```

---

## Git

### Push changes
```bash
git add .
git commit -m "your message"
git push
```

### Check CI/CD pipeline
```bash
# Go to GitHub → Actions tab
# Every push triggers ml_pipeline.yml automatically
```

---

## Quick Deploy Script
```bash
# Run this to redeploy from scratch after cluster deletion

gcloud container clusters create finance-ai-cluster \
  --zone us-central1-a \
  --num-nodes 2 \
  --machine-type e2-medium \
  --disk-size 20GB

gcloud container clusters get-credentials finance-ai-cluster \
  --zone us-central1-a

kubectl create secret generic finance-ai-secrets \
  --from-literal=anthropic-api-key=$(grep ANTHROPIC_API_KEY .env | cut -d '=' -f2) \
  --from-literal=langchain-api-key=$(grep LANGCHAIN_API_KEY .env | cut -d '=' -f2)

kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

kubectl scale deployment/finance-ai-api --replicas=1

echo "Waiting for external IP..."
kubectl get service finance-ai-service --watch
```