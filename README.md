# Containerized Model Deployer (CMD)

## Overview
The Containerized Model Deployer (CMD) project is designed to build and deploy machine learning models in a scalable, automated manner using Docker, Kubernetes, and TensorFlow Extended (TFX). The system provides an end-to-end solution for model deployment, serving, and efficient resource management within Kubernetes.

## Key Features
- **Docker**: Containerizes machine learning models to ensure portability and consistency across different environments.
- **Kubernetes**: Orchestrates the deployment, scaling, and management of model containers to guarantee high availability.
- **TensorFlow Extended (TFX)**: Manages the machine learning pipeline, including model training, evaluation, and deployment.

## Project Structure
containerized-model-deployer/
├── tfx_pipeline/
│   ├── __init__.py
│   ├── data/                # Raw and preprocessed data
│   │   ├── train_data.csv
│   │   ├── test_data.csv
│   ├── components/          # Custom components for the TFX pipeline
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   ├── trainer.py
│   ├── pipeline/            # TFX pipeline definition
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   ├── models/              # Saved models
│       ├── saved_model/
├── docker/
│   ├── Dockerfile           # Dockerfile to containerize the app
│   ├── entrypoint.sh        # Entrypoint script for the container
├── k8s/
│   ├── deployment.yaml      # Kubernetes deployment configuration
│   ├── service.yaml         # Kubernetes service configuration
├── scripts/
│   ├── build_docker.sh      # Script to build the Docker image
│   ├── deploy_k8s.sh        # Script to deploy to Kubernetes
│   ├── test_container.sh    # Script to test the Docker container locally
├── README.md                # Documentation
├── .gitignore               # Git ignore file




## Prerequisites
Before running the project, ensure that the following tools are installed:
- **Docker**: To containerize the application.
- **Kubernetes**: A Kubernetes cluster (Minikube, GKE, or similar) to deploy and manage the containerized application.
- **TensorFlow Extended (TFX)**: To manage and build the machine learning pipeline.
  
## Getting Started

### 1. Clone the Repository
Clone the repository to your local machine.

```bash
git clone https://github.com/your-repo/containerized-model-deployer.git
cd containerized-model-deployer

