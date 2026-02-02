"""
Command-line argument parser for wisent.

This module sets up the main CLI parser with all subcommands.
Parser setup functions are imported from wisent.core.parser_arguments.
"""

import argparse
from typing import List, Optional

from wisent.core.errors import ModelNotProvidedError, InvalidValueError

# Import all setup functions from parser_arguments
from wisent.core.parser_arguments.tasks_parser import setup_tasks_parser
from wisent.core.parser_arguments.generate_pairs_parser import setup_generate_pairs_parser
from wisent.core.parser_arguments.synthetic_parser import setup_synthetic_parser
from wisent.core.parser_arguments.nonsense_parser import setup_test_nonsense_parser
from wisent.core.parser_arguments.monitor_parser import setup_monitor_parser
from wisent.core.parser_arguments.agent_parser import setup_agent_parser
from wisent.core.parser_arguments.model_config_parser import setup_model_config_parser
from wisent.core.parser_arguments.configure_model_parser import setup_configure_model_parser
from wisent.core.parser_arguments.optimize_classification_parser import setup_classification_optimizer_parser
from wisent.core.parser_arguments.optimize_steering_parser import setup_steering_optimizer_parser
from wisent.core.parser_arguments.optimize_sample_size_parser import setup_sample_size_optimizer_parser
from wisent.core.parser_arguments.full_optimize_parser import setup_full_optimizer_parser
from wisent.core.parser_arguments.optimize_parser import setup_optimize_parser
from wisent.core.parser_arguments.generate_vector_parser import setup_generate_vector_parser
from wisent.core.parser_arguments.multi_steer_parser import setup_multi_steer_parser
from wisent.core.parser_arguments.evaluate_parser import setup_evaluate_parser
from wisent.core.parser_arguments.train_unified_goodness_parser import setup_train_unified_goodness_parser
from wisent.core.parser_arguments.repscan_parser import setup_repscan_parser


def setup_parser() -> argparse.ArgumentParser:
    """Set up the main CLI parser with subcommands."""
    parser = argparse.ArgumentParser(description="Wisent: Advanced AI Safety and Alignment Toolkit")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Core commands
    tasks_parser = subparsers.add_parser("tasks", help="Run evaluation tasks")
    setup_tasks_parser(tasks_parser)

    generate_parser = subparsers.add_parser("generate-pairs", help="Generate synthetic contrastive pairs")
    setup_generate_pairs_parser(generate_parser)

    synthetic_parser = subparsers.add_parser("synthetic", help="Run synthetic contrastive pair pipeline")
    setup_synthetic_parser(synthetic_parser)

    test_nonsense_parser = subparsers.add_parser("test-nonsense", help="Test nonsense detection system")
    setup_test_nonsense_parser(test_nonsense_parser)

    monitor_parser = subparsers.add_parser("monitor", help="Performance monitoring and system information")
    setup_monitor_parser(monitor_parser)

    agent_parser = subparsers.add_parser("agent", help="Interact with autonomous agent")
    setup_agent_parser(agent_parser)

    # Configuration commands
    model_config_parser = subparsers.add_parser("model-config", help="Manage model-specific optimal parameters")
    setup_model_config_parser(model_config_parser)

    configure_model_parser = subparsers.add_parser(
        "configure-model", help="Configure tokens and layer access for unsupported models"
    )
    setup_configure_model_parser(configure_model_parser)

    # Optimization commands
    classification_optimizer_parser = subparsers.add_parser(
        "optimize-classification", help="Optimize classification parameters across all tasks"
    )
    setup_classification_optimizer_parser(classification_optimizer_parser)

    steering_optimizer_parser = subparsers.add_parser(
        "optimize-steering", help="Optimize steering parameters for different methods"
    )
    setup_steering_optimizer_parser(steering_optimizer_parser)

    sample_size_optimizer_parser = subparsers.add_parser(
        "optimize-sample-size", help="Find optimal training sample size for classifiers"
    )
    setup_sample_size_optimizer_parser(sample_size_optimizer_parser)

    full_optimizer_parser = subparsers.add_parser(
        "full-optimize", help="Run full optimization: classification parameters then sample size"
    )
    setup_full_optimizer_parser(full_optimizer_parser)

    optimize_parser = subparsers.add_parser(
        "optimize", help="Full Optuna-based optimization: classification + steering (ALL methods) + weights"
    )
    setup_optimize_parser(optimize_parser)

    # Vector and steering commands
    generate_vector_parser = subparsers.add_parser(
        "generate-vector", help="Generate steering vectors from contrastive pairs (file or description)"
    )
    setup_generate_vector_parser(generate_vector_parser)

    multi_steer_parser = subparsers.add_parser(
        "multi-steer", help="Combine multiple steering vectors dynamically at inference time"
    )
    setup_multi_steer_parser(multi_steer_parser)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate single prompt with steering vector and return quality scores"
    )
    setup_evaluate_parser(evaluate_parser)

    # Training commands
    unified_goodness_parser = subparsers.add_parser(
        "train-unified-goodness",
        help="Train a single 'goodness' steering vector from pooled multi-benchmark data"
    )
    setup_train_unified_goodness_parser(unified_goodness_parser)

    # Analysis commands
    repscan_parser = subparsers.add_parser(
        "repscan", help="Run RepScan geometry analysis with concept decomposition on database activations"
    )
    setup_repscan_parser(repscan_parser)

    return parser


def validate_args(args) -> None:
    """Validate parsed arguments."""
    if args.command == "tasks":
        if not hasattr(args, 'model') or not args.model:
            if not (getattr(args, 'list_tasks', False) or getattr(args, 'task_info', None)):
                raise ModelNotProvidedError("--model is required for running tasks")

        if hasattr(args, 'layer') and args.layer:
            try:
                if '-' in str(args.layer):
                    start, end = map(int, str(args.layer).split('-'))
                    if start >= end:
                        raise InvalidValueError(f"Invalid layer range: {args.layer}")
                elif ',' in str(args.layer):
                    [int(l.strip()) for l in str(args.layer).split(',')]
                else:
                    int(args.layer)
            except ValueError:
                raise InvalidValueError(f"Invalid layer specification: {args.layer}")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = setup_parser()
    parsed = parser.parse_args(args)
    validate_args(parsed)
    return parsed
