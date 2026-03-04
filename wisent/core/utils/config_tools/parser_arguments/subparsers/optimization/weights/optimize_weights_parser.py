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


from wisent.core.utils.config_tools.parser_arguments.optimization.weights.optimize_weights_parser_advanced import (
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
            "(3) Built-in example e.g. 'wisent.core.reading.evaluators.custom.examples.gptzero'"
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
        required=True,
        help="Fraction of pairs for training vs evaluation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: 42)"
    )

    # Personalization evaluator quality parameters
    parser.add_argument(
        "--fast-diversity-seed", type=int, required=True,
        help="Seed for fast diversity computation"
    )
    parser.add_argument(
        "--diversity-max-sample-size", type=int, required=True,
        help="Maximum sample size for diversity computation"
    )
    parser.add_argument(
        "--min-sentence-length", type=int, required=True,
        help="Minimum sentence length for coherence evaluation"
    )
    parser.add_argument(
        "--nonsense-min-tokens", type=int, required=True,
        help="Minimum token count for nonsense word detection"
    )
    parser.add_argument(
        "--quality-min-response-length", type=int, required=True,
        help="Minimum response length for quality scoring"
    )
    parser.add_argument(
        "--quality-repetition-ratio-threshold", type=float, required=True,
        help="Threshold for repetitive token ratio penalty"
    )
    parser.add_argument(
        "--quality-bigram-repeat-threshold", type=int, required=True,
        help="Threshold for repeated bigram count penalty"
    )
    parser.add_argument(
        "--quality-bigram-repeat-penalty", type=float, required=True,
        help="Penalty multiplier for repeated bigrams"
    )
    parser.add_argument(
        "--quality-special-char-ratio-threshold", type=float, required=True,
        help="Threshold for special character ratio penalty"
    )
    parser.add_argument(
        "--quality-special-char-penalty", type=float, required=True,
        help="Penalty multiplier for excessive special characters"
    )
    parser.add_argument(
        "--quality-char-repeat-count", type=int, required=True,
        help="Minimum consecutive character repeats to trigger penalty"
    )
    parser.add_argument(
        "--quality-char-repeat-penalty", type=float, required=True,
        help="Penalty multiplier for character repetition"
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
        required=True,
        help="Save checkpoint and best model every N trials"
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
        required=True,
        help="Number of evaluation prompts per trial"
    )

    # ==========================================================================
    # OPTIMIZATION TARGET
    # ==========================================================================
    target_group = parser.add_argument_group("optimization target")
    target_group.add_argument(
        "--target-metric",
        type=str,
        required=True,
        help="Metric to optimize: compliance_rate, refusal_rate, accuracy"
    )
    target_group.add_argument(
        "--target-value",
        type=float,
        required=True,
        help="Target value for the metric"
    )
    target_group.add_argument(
        "--direction",
        type=str,
        required=True,
        choices=["auto", "maximize", "minimize"],
        help="Optimization direction. 'auto' infers from metric"
    )

    setup_advanced_optimize_weights_args(parser)
