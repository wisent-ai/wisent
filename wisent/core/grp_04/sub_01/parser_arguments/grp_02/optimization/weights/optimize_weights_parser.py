"""
Parser for optimize-weights command.

This command runs an optimization loop to find optimal weight modification
parameters for any task type.

Unified pipeline:
1. Generate steering vector based on --task type
2. Apply weight modification with trial parameters
3. Evaluate using task-appropriate evaluator
4. Optuna adjusts parameters
5. Repeat until target reached or max trials
6. Save optimized model

Task types:
- refusal: Compliance rate optimization
- personalization: Personality steering (requires --trait)
- custom: Custom evaluator (requires --custom-evaluator)
- benchmark: Single benchmark accuracy (e.g., arc_easy)
- multi-benchmark: Comma-separated benchmarks (e.g., arc_easy,gsm8k)
"""

import argparse

from wisent.core.constants import (
    DATA_SPLIT_RATIO,
    DATA_SPLIT_SEED,
    OPTIMIZE_WEIGHTS_CHECKPOINT_INTERVAL,
    OPTIMIZE_WEIGHTS_NUM_EVAL_PROMPTS,
    OPTIMIZE_WEIGHTS_TARGET_VALUE,
)

from wisent.core.parser_arguments.optimization.weights.optimize_weights_parser_advanced import (
    setup_advanced_optimize_weights_args,
)


def setup_optimize_weights_parser(parser: argparse.ArgumentParser) -> None:
    """
    Set up argument parser for optimize-weights command.

    This command optimizes weight modification parameters to achieve a target
    metric value. Works with any --task type (refusal, personalization, 
    single benchmark, or comma-separated benchmarks).
    """

    # ==========================================================================
    # INPUT SOURCE: What to optimize for
    # ==========================================================================
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--task",
        type=str,
        help=(
            "Task to optimize for. Can be: "
            "'refusal' (compliance optimization), "
            "'personalization' (requires --trait), "
            "'custom' (requires --custom-evaluator), "
            "benchmark name (e.g., 'arc_easy', 'gsm8k'), "
            "or comma-separated benchmarks (e.g., 'arc_easy,gsm8k,hellaswag')"
        )
    )
    input_group.add_argument(
        "--steering-vectors",
        type=str,
        help="Path to pre-computed steering vectors JSON file (skip vector generation)"
    )
    
    # Trait description (required for --task personalization)
    parser.add_argument(
        "--trait",
        type=str,
        default=None,
        help="Trait description for personalization (required when --task personalization)"
    )
    
    # Custom evaluator (required for --task custom)
    parser.add_argument(
        "--custom-evaluator",
        type=str,
        default=None,
        help=(
            "Custom evaluator specification (required when --task custom). Can be: "
            "(1) Python module path e.g. 'my_evaluators.gptzero', "
            "(2) File path with function e.g. './my_eval.py:score_fn', "
            "(3) Built-in example e.g. 'wisent.core.evaluators.custom.examples.gptzero'"
        )
    )
    parser.add_argument(
        "--custom-evaluator-kwargs",
        type=str,
        default=None,
        help="JSON string of kwargs for custom evaluator, e.g. '{\"api_key\": \"xxx\"}'"
    )
    
    # Additional options for multi-benchmark mode (--task bench1,bench2,...)
    parser.add_argument(
        "--cap-pairs-per-benchmark",
        type=int,
        default=None,
        help="Cap pairs per benchmark. Benchmarks with more pairs get randomly sampled."
    )
    parser.add_argument(
        "--max-benchmarks",
        type=int,
        default=None,
        help="Maximum number of benchmarks to use"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DATA_SPLIT_RATIO,
        help="Fraction of pairs for training vs evaluation (default: 0.8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DATA_SPLIT_SEED,
        help="Random seed for reproducibility (default: 42)"
    )

    # ==========================================================================
    # MODEL CONFIGURATION
    # ==========================================================================
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cuda', 'cuda:0', 'cpu'). Default: auto"
    )

    # ==========================================================================
    # OUTPUT
    # ==========================================================================
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the optimized model"
    )
    parser.add_argument(
        "--save-trials",
        type=str,
        default=None,
        help="Path to save all trial results as JSON (optional)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file for saving/resuming optimization. If file exists, resume from it."
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=OPTIMIZE_WEIGHTS_CHECKPOINT_INTERVAL,
        help="Save checkpoint and best model every N trials. Default: 5"
    )
    parser.add_argument(
        "--gcs-bucket",
        type=str,
        default=None,
        help="GCS bucket to upload results to (e.g., 'wisent-optimization-results'). Results will be uploaded on completion."
    )

    # ==========================================================================
    # EVALUATION CONFIGURATION
    # ==========================================================================
    eval_group = parser.add_argument_group("evaluation")
    eval_group.add_argument(
        "--eval-prompts",
        type=str,
        default=None,
        help="Path to custom evaluation prompts JSON file"
    )
    eval_group.add_argument(
        "--eval-topics",
        type=str,
        default=None,
        help="Comma-separated UncensorBench topics (for --task refusal)"
    )
    eval_group.add_argument(
        "--num-eval-prompts",
        type=int,
        default=OPTIMIZE_WEIGHTS_NUM_EVAL_PROMPTS,
        help="Number of evaluation prompts per trial. Default: 30"
    )

    # ==========================================================================
    # OPTIMIZATION TARGET
    # ==========================================================================
    target_group = parser.add_argument_group("optimization target")
    target_group.add_argument(
        "--target-metric",
        type=str,
        default="compliance_rate",
        help="Metric to optimize: compliance_rate, refusal_rate, accuracy. Default: compliance_rate"
    )
    target_group.add_argument(
        "--target-value",
        type=float,
        default=OPTIMIZE_WEIGHTS_TARGET_VALUE,
        help="Target value for the metric. Default: 0.95"
    )
    target_group.add_argument(
        "--direction",
        type=str,
        default="auto",
        choices=["auto", "maximize", "minimize"],
        help="Optimization direction. 'auto' infers from metric. Default: auto"
    )

    setup_advanced_optimize_weights_args(parser)
