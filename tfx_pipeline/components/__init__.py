"""
Initialization file for the components module.

This module provides the core functionality for preprocessing and training
within the TFX pipeline.
"""

from tfx_pipeline.components import preprocess_data, train_model


# Expose these functions for easier imports
__all__ = ["preprocess_data", "train_model"]

