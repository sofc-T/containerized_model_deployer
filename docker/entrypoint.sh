# Run the TFX pipeline
python -m tfx_pipeline.pipeline.pipeline

# (Optional) Start TensorFlow Serving for deployed models
tensorflow_model_server --rest_api_port=8501 \
    --model_base_path="/app/tfx_pipeline/models/saved_model" \
    --model_name=tfx_model
