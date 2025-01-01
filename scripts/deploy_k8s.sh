# Apply the deployment and service configuration to Kubernetes
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Optional: If using Horizontal Pod Autoscaler
kubectl apply -f k8s/hpa.yaml
