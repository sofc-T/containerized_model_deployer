"""
Initialization module for the TFX pipeline.

This module exposes the key pipeline components for external use.
"""

from .pipeline import create_pipeline, run_pipeline

__all__ = ["create_pipeline", "run_pipeline"]
