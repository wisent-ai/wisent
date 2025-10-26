"""
Main parser setup for Wisent CLI.

This module imports and combines all command-specific parsers into a single
argparse parser for the Wisent CLI.
"""

import argparse

from wisent.core.parser_arguments.tasks_parser import setup_tasks_parser
from wisent.core.parser_arguments.generate_pairs_parser import setup_generate_pairs_parser
from wisent.core.parser_arguments.generate_pairs_from_task_parser import setup_generate_pairs_from_task_parser
from wisent.core.parser_arguments.get_activations_parser import setup_get_activations_parser
from wisent.core.parser_arguments.create_steering_vector_parser import setup_create_steering_vector_parser
from wisent.core.parser_arguments.generate_vector_from_task_parser import setup_generate_vector_from_task_parser
from wisent.core.parser_arguments.generate_vector_from_synthetic_parser import setup_generate_vector_from_synthetic_parser
from wisent.core.parser_arguments.synthetic_parser import setup_synthetic_parser
from wisent.core.parser_arguments.test_nonsense_parser import setup_test_nonsense_parser
from wisent.core.parser_arguments.monitor_parser import setup_monitor_parser
from wisent.core.parser_arguments.agent_parser import setup_agent_parser
from wisent.core.parser_arguments.model_config_parser import setup_model_config_parser
from wisent.core.parser_arguments.configure_model_parser import setup_configure_model_parser
from wisent.core.parser_arguments.optimize_classification_parser import setup_classification_optimizer_parser
from wisent.core.parser_arguments.optimize_steering_parser import setup_steering_optimizer_parser
from wisent.core.parser_arguments.optimize_sample_size_parser import setup_sample_size_optimizer_parser
from wisent.core.parser_arguments.full_optimize_parser import setup_full_optimizer_parser
from wisent.core.parser_arguments.generate_vector_parser import setup_generate_vector_parser
from wisent.core.parser_arguments.multi_steer_parser import setup_multi_steer_parser
from wisent.core.parser_arguments.evaluate_parser import setup_evaluate_parser


def setup_parser() -> argparse.ArgumentParser:
    """Set up the main CLI parser with subcommands."""
    parser = argparse.ArgumentParser(description="Wisent-Guard: Advanced AI Safety and Alignment Toolkit")

    # Global arguments
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Tasks command (main evaluation pipeline)
    tasks_parser = subparsers.add_parser("tasks", help="Run evaluation tasks")
    setup_tasks_parser(tasks_parser)

    # Generate pairs command
    generate_parser = subparsers.add_parser("generate-pairs", help="Generate synthetic contrastive pairs")
    setup_generate_pairs_parser(generate_parser)

    # Generate pairs from task command
    generate_from_task_parser = subparsers.add_parser("generate-pairs-from-task", help="Generate contrastive pairs from lm-eval task")

    # Get activations command
    get_activations_parser = subparsers.add_parser("get-activations", help="Collect activations from contrastive pairs")
    setup_get_activations_parser(get_activations_parser)
    setup_generate_pairs_from_task_parser(generate_from_task_parser)

    # Create steering vector command
    create_steering_parser = subparsers.add_parser("create-steering-vector", help="Create steering vectors from enriched pairs")
    setup_create_steering_vector_parser(create_steering_parser)

    # Generate vector from task command (full pipeline)
    generate_vector_from_task_parser = subparsers.add_parser("generate-vector-from-task", help="Generate steering vector from task (full pipeline)")
    setup_generate_vector_from_task_parser(generate_vector_from_task_parser)

    # Generate vector from synthetic command (full pipeline)
    generate_vector_from_synthetic_parser = subparsers.add_parser("generate-vector-from-synthetic", help="Generate steering vector from synthetic pairs (full pipeline)")
    setup_generate_vector_from_synthetic_parser(generate_vector_from_synthetic_parser)

    # Synthetic command (generate + train + test)
    synthetic_parser = subparsers.add_parser("synthetic", help="Run synthetic contrastive pair pipeline")
    setup_synthetic_parser(synthetic_parser)

    # Test nonsense detection command
    test_nonsense_parser = subparsers.add_parser("test-nonsense", help="Test nonsense detection system")
    setup_test_nonsense_parser(test_nonsense_parser)

    # Monitor command for performance monitoring
    monitor_parser = subparsers.add_parser("monitor", help="Performance monitoring and system information")
    setup_monitor_parser(monitor_parser)

    # Agent command for autonomous agent interaction
    agent_parser = subparsers.add_parser("agent", help="Interact with autonomous agent")
    setup_agent_parser(agent_parser)

    # Model configuration command for managing optimal parameters
    model_config_parser = subparsers.add_parser("model-config", help="Manage model-specific optimal parameters")
    setup_model_config_parser(model_config_parser)

    # Configure model command for setting up new/unsupported models
    configure_model_parser = subparsers.add_parser(
        "configure-model", help="Configure tokens and layer access for unsupported models"
    )
    setup_configure_model_parser(configure_model_parser)

    # Classification optimization command for finding optimal classification parameters
    classification_optimizer_parser = subparsers.add_parser(
        "optimize-classification", help="Optimize classification parameters across all tasks"
    )
    setup_classification_optimizer_parser(classification_optimizer_parser)

    # Steering optimization command for finding optimal steering parameters
    steering_optimizer_parser = subparsers.add_parser(
        "optimize-steering", help="Optimize steering parameters for different methods"
    )
    setup_steering_optimizer_parser(steering_optimizer_parser)

    # Sample size optimization command for finding optimal training sample sizes
    sample_size_optimizer_parser = subparsers.add_parser(
        "optimize-sample-size", help="Find optimal training sample size for classifiers"
    )
    setup_sample_size_optimizer_parser(sample_size_optimizer_parser)

    # Full optimization command that runs both classification and sample size optimization
    full_optimizer_parser = subparsers.add_parser(
        "full-optimize", help="Run full optimization: classification parameters then sample size"
    )
    setup_full_optimizer_parser(full_optimizer_parser)

    # Generate vector command for creating steering vectors without tasks
    generate_vector_parser = subparsers.add_parser(
        "generate-vector", help="Generate steering vectors from contrastive pairs (file or description)"
    )
    setup_generate_vector_parser(generate_vector_parser)

    # Multi-vector steering command for combining multiple vectors at inference time
    multi_steer_parser = subparsers.add_parser(
        "multi-steer", help="Combine multiple steering vectors dynamically at inference time"
    )
    setup_multi_steer_parser(multi_steer_parser)

    # Single-prompt evaluation command for real-time steering assessment
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate single prompt with steering vector and return quality scores"
    )
    setup_evaluate_parser(evaluate_parser)

    return parser
