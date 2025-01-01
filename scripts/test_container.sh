#!/bin/bash

# Build the Docker image
docker build -t "docker-image"

# Run the Docker container locally
docker run -p 8501:8501 "docker-image"

# Test the model endpoint locally (after running the container)
curl -d '{"instances": [[1.0, 2.0, 3.0]]}' \
    -H "Content-Type: application/json" \
    -X POST http://localhost:8501/v1/models/tfx_model:predict
