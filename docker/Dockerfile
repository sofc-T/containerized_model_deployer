# Use the official TensorFlow Extended base image
FROM tensorflow/tfx:latest

# Set the working directory
WORKDIR /app

# Copy the pipeline code and configuration
COPY tfx_pipeline /app/tfx_pipeline
COPY docker/entrypoint.sh /app/entrypoint.sh

# Install additional dependencies (if any)
RUN apt-get update && apt-get install -y \
    && pip install --no-cache-dir \
       tensorflow-serving-api \
       pandas \
       numpy \
    && chmod +x /app/entrypoint.sh

# Expose the TensorFlow Serving default port
EXPOSE 8501

# Set the entry point for the container
ENTRYPOINT ["/app/entrypoint.sh"]
