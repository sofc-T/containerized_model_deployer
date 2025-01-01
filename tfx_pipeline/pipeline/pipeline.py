"""
Pipeline module for the TFX workflow.

This module defines the TFX pipeline, including all the components
required for data preprocessing, model training, and evaluation.
"""

import os
from tfx.orchestration import pipeline
from tfx.components import CsvExampleGen, Trainer, Evaluator, Pusher
from tfx.proto import trainer_pb2
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.dsl.components.common.resolver import Resolver
from tfx.components.common.nodes.resolver_node import ResolverStrategy

# Define paths
_pipeline_root = os.path.join(os.getcwd(), "tfx_pipeline")
_data_path = os.path.join(_pipeline_root, "data")
_model_dir = os.path.join(_pipeline_root, "models")
_serving_dir = os.path.join(_pipeline_root, "serving_model")

def create_pipeline(pipeline_name="CMD_pipeline"):
    """
    Creates a TFX pipeline.

    Args:
        pipeline_name (str): Name of the pipeline.

    Returns:
        tfx.orchestration.pipeline.Pipeline: A configured TFX pipeline.
    """
    # Data ingestion: Reads data from CSV files
    example_gen = CsvExampleGen(input_base=_data_path)

    # Model trainer
    trainer = Trainer(
        module_file=os.path.join(_pipeline_root, "components/trainer.py"),
        custom_executor_spec=None,
        examples=example_gen.outputs["examples"],
        train_args=trainer_pb2.TrainArgs(num_steps=200),
        eval_args=trainer_pb2.EvalArgs(num_steps=50),
    )

    # Evaluator: Compares the current model with the previous one
    evaluator = Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
    )

    # Pusher: Pushes the trained model to a serving directory
    pusher = Pusher(
        model=trainer.outputs["model"],
        push_destination=pipeline_pb2.PushDestination(
            filesystem=pipeline_pb2.PushDestination.Filesystem(
                base_directory=_serving_dir
            )
        ),
    )

    # Define the pipeline
    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=_pipeline_root,
        components=[
            example_gen,
            trainer,
            evaluator,
            pusher,
        ],
    )

def run_pipeline(pipeline_instance):
    """
    Runs the TFX pipeline.

    Args:
        pipeline_instance (tfx.orchestration.pipeline.Pipeline): Configured TFX pipeline instance.
    """
    runner = LocalDagRunner()
    runner.run(pipeline_instance)

if __name__ == "__main__":
    # Create the pipeline
    pipeline_instance = create_pipeline()

    # Run the pipeline
    run_pipeline(pipeline_instance)
