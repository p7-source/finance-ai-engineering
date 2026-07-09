#!/bin/bash
# start.sh
# Spins up full Finance AI system on GKE
# Use before interviews/demos

echo "=== STARTING FINANCE AI CLUSTER ==="
echo "⏱️  Takes about 5 minutes total"
echo ""

# Set project
echo "Step 1: Setting GCP project..."
gcloud config set project mlops-lab-prassadh-2026
echo "✅ Project set"

# Build and push image
echo ""
echo "Step 2: Building and pushing Docker image..."
docker buildx build \
  --platform linux/amd64 \
  -t gcr.io/mlops-lab-prassadh-2026/finance-ai-model:latest \
  --push \
  .
echo "✅ Image pushed to GCR"

# Create cluster
echo ""
echo "Step 3: Creating GKE cluster..."
gcloud container clusters create finance-ai-cluster \
  --zone us-central1-a \
  --num-nodes 2 \
  --machine-type e2-medium \
  --disk-size 20GB \
  --quiet
echo "✅ Cluster created"

# Connect kubectl
echo ""
echo "Step 4: Connecting kubectl..."
gcloud container clusters get-credentials finance-ai-cluster \
  --zone us-central1-a
echo "✅ kubectl connected"

# Create secrets
echo ""
echo "Step 5: Creating secrets..."
kubectl create secret generic finance-ai-secrets \
  --from-literal=anthropic-api-key=$(grep ANTHROPIC_API_KEY .env | cut -d '=' -f2) \
  --from-literal=langchain-api-key=$(grep LANGCHAIN_API_KEY .env | cut -d '=' -f2)
echo "✅ Secrets created"

# Deploy manifests
echo ""
echo "Step 6: Deploying to Kubernetes..."
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
echo "✅ Manifests applied"

# Scale to 1 replica
echo ""
echo "Step 7: Scaling deployment..."
kubectl scale deployment/finance-ai-api --replicas=1
echo "✅ Scaled to 1 replica"

# Wait for pod to be ready
echo ""
echo "Step 8: Waiting for pod to be ready..."
kubectl wait --for=condition=ready pod \
  -l app=finance-ai-api \
  --timeout=120s
echo "✅ Pod is ready"

# Get external IP
echo ""
echo "Step 9: Getting external IP..."
echo "⏱️  Waiting for LoadBalancer IP (30 seconds)..."
sleep 30
EXTERNAL_IP=$(kubectl get service finance-ai-service \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "✅ External IP: $EXTERNAL_IP"

# Test the API
echo ""
echo "Step 10: Testing live API..."
curl -s http://$EXTERNAL_IP/health
echo ""

echo ""
echo "=== FINANCE AI SYSTEM IS LIVE ==="
echo "🚀 External IP: $EXTERNAL_IP"
echo ""
echo "Test commands:"
echo "curl http://$EXTERNAL_IP/health"
echo "curl -X POST http://$EXTERNAL_IP/query -H 'Content-Type: application/json' -d '{\"question\": \"What is Basel III?\"}'"
echo ""
echo "💰 Remember to run ./destroy.sh after demo to stop billing"