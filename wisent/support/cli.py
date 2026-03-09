"""
Programmatic API for running Wisent tasks.

This module provides a Python API for running tasks that would normally be invoked via CLI.
"""

import argparse
from typing import Dict, Any, Optional
from wisent.core.utils.cli import execute_tasks
from wisent.core.utils.config_tools.constants import DEFAULT_RANDOM_SEED


def run_task_pipeline(
    task_name: str,
    model_name: str,
    layer: str,
    token_aggregation: str,
    classifier_type: str,
    token_targeting_strategy: str,
    steering_method: Optional[str] = None,
    seed: int = DEFAULT_RANDOM_SEED,
    verbose: bool = False,
    split_ratio: float = 0.8,
    steering_mode: bool = False,
    detection_threshold: float = None,
    steering_strength: Optional[float] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run a task pipeline programmatically.

    This function provides a programmatic interface to the tasks command,
    allowing other modules to run tasks without invoking the CLI.

    Args:
        task_name: Name of the task to run
        model_name: Name or path of the model
        layer: Layer to use for activations
        seed: Random seed for reproducibility
        verbose: Whether to print verbose output
        split_ratio: Train/test split ratio
        steering_mode: Whether to use steering mode
        token_aggregation: Token aggregation strategy
        detection_threshold: Detection threshold for classification
        classifier_type: Type of classifier to use
        steering_method: Steering method to use (if steering_mode=True)
        steering_strength: Steering strength (if steering_mode=True)
        token_targeting_strategy: Token targeting strategy for steering
        **kwargs: Additional arguments

    Returns:
        Dictionary containing results including:
        - test_accuracy: Test accuracy
        - test_f1_score: Test F1 score
        - training_time: Time spent training
        - evaluation_results: Evaluation results dict
    """
    # Create a namespace object that mimics argparse output
    args = argparse.Namespace()

    # Set required arguments
    # task_name is already a string, keep it as a list
    args.task_names = [task_name] if isinstance(task_name, str) else task_name
    args.model = model_name
    args.layer = int(layer) if isinstance(layer, str) else layer
    args.seed = seed
    args.verbose = verbose
    args.split_ratio = split_ratio

    # Set method-specific arguments
    if steering_mode:
        if steering_method is None:
            from wisent.core.utils.infra_tools.errors import MissingParameterError
            raise MissingParameterError(
                params=["steering_method"],
                context="run_task_pipeline with steering_mode=True",
            )
        args.steering_mode = True
        args.steering_method = steering_method
        args.steering_strength = steering_strength
        args.token_targeting_strategy = token_targeting_strategy
        args.token_aggregation = token_aggregation
    else:
        if detection_threshold is None:
            raise ValueError("detection_threshold is required when steering_mode=False")
        args.steering_mode = False
        args.token_aggregation = token_aggregation
        args.detection_threshold = detection_threshold
        args.classifier_type = classifier_type

    # Set defaults for other arguments
    args.train_only = False
    args.inference_only = False
    args.save_classifier = None
    args.load_classifier = None
    args.save_steering_vector = None
    args.load_steering_vector = None
    args.output = None
    args.evaluation_report = None
    args.device = kwargs.get('device', 'cpu')

    # Execute the task and capture results
    results = execute_tasks(args)

    # If execute_tasks returns None (shouldn't happen with our changes, but handle it)
    if results is None:
        return {
            "test_accuracy": 0.0,
            "test_f1_score": 0.0,
            "training_time": 0.0,
            "evaluation_results": {}
        }

    return results
