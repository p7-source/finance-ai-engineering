#!/bin/bash
# destroy.sh
# Tears down all GKE resources to save cost
# Run this after demos/interviews

echo "=== DESTROYING FINANCE AI CLUSTER ==="
echo "⚠️  This will delete all resources and stop billing"
echo ""

# Delete Kubernetes resources
echo "Step 1: Deleting Kubernetes resources..."
kubectl delete -f k8s/ --ignore-not-found
kubectl delete secret finance-ai-secrets --ignore-not-found
echo "✅ Kubernetes resources deleted"

# Delete GKE cluster
echo ""
echo "Step 2: Deleting GKE cluster..."
gcloud container clusters delete finance-ai-cluster \
  --zone us-central1-a \
  --quiet
echo "✅ GKE cluster deleted"

# Delete Docker image from GCR
echo ""
echo "Step 3: Deleting Docker image from GCR..."
gcloud container images delete \
  gcr.io/mlops-lab-prassadh-2026/finance-ai-model:latest \
  --quiet --force-delete-tags
echo "✅ Docker image deleted"

echo ""
echo "=== DESTROY COMPLETE ==="
echo "💰 Billing stopped — no more charges"
echo ""
echo "To redeploy: copy commands from COMMANDS.md"