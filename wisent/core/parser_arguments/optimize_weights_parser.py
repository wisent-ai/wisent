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
        default=0.8,
        help="Fraction of pairs for training vs evaluation (default: 0.8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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
        default=5,
        help="Save checkpoint and best model every N trials. Default: 5"
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=None,
        help="S3 bucket to upload results to (e.g., 'wisent-optimization-results'). Results will be uploaded on completion."
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
        default=30,
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
        default=0.95,
        help="Target value for the metric. Default: 0.95"
    )
    target_group.add_argument(
        "--direction",
        type=str,
        default="auto",
        choices=["auto", "maximize", "minimize"],
        help="Optimization direction. 'auto' infers from metric. Default: auto"
    )

    # ==========================================================================
    # OPTIMIZATION PARAMETERS
    # ==========================================================================
    optim_group = parser.add_argument_group("optimization")
    optim_group.add_argument(
        "--trials",
        type=int,
        default=300,
        help="Number of optimization trials. Default: 300"
    )
    optim_group.add_argument(
        "--startup-trials",
        type=int,
        default=10,
        help="Number of random startup trials before TPE. Default: 10"
    )
    optim_group.add_argument(
        "--early-stop",
        action="store_true",
        help="Stop early if target value is reached"
    )
    optim_group.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
        help="Stop if no improvement for N trials. Default: 10"
    )

    # ==========================================================================
    # SEARCH SPACE: Parameters to optimize
    # ==========================================================================
    search_group = parser.add_argument_group("search space")
    search_group.add_argument(
        "--strength-range",
        type=str,
        default="0.3,2.0",
        help="Min,max range for directional projection strength. Default: 0.3,2.0"
    )
    search_group.add_argument(
        "--max-weight-range",
        type=str,
        default="0.5,3.0",
        help="Min,max range for kernel max weight. Default: 0.5,3.0"
    )
    search_group.add_argument(
        "--min-weight-range",
        type=str,
        default="0.0,0.5",
        help="Min,max range for kernel min weight. Default: 0.0,0.5"
    )
    search_group.add_argument(
        "--position-range",
        type=str,
        default="0.3,0.7",
        help="Min,max range for kernel peak position (as ratio 0-1). Default: 0.3,0.7"
    )
    search_group.add_argument(
        "--num-pairs",
        type=int,
        default=100,
        help="Number of contrastive pairs to generate (fixed, not optimized). Default: 100"
    )
    search_group.add_argument(
        "--optimize-direction-index",
        action="store_true",
        help="Optimize float direction index (interpolate between layer directions)"
    )

    # ==========================================================================
    # WEIGHT MODIFICATION METHOD
    # ==========================================================================
    method_group = parser.add_argument_group("weight modification")
    method_group.add_argument(
        "--method",
        type=str,
        default="directional",
        choices=["directional", "additive", "titan", "prism", "pulse"],
        help=(
            "Weight modification method. Default: directional. "
            "Options: directional (single direction), additive (bias-based), "
            "titan/prism/pulse (multi-direction, better for non-linear representations)"
        )
    )
    method_group.add_argument(
        "--components",
        type=str,
        nargs="+",
        default=["self_attn.o_proj", "mlp.down_proj"],
        help="Components to modify. Default: self_attn.o_proj mlp.down_proj"
    )
    method_group.add_argument(
        "--norm-preserve",
        action="store_true",
        default=True,
        help="Use norm-preserving directional projection. Default: True"
    )
    method_group.add_argument(
        "--no-norm-preserve",
        action="store_false",
        dest="norm_preserve",
        help="Disable norm-preserving directional projection"
    )
    
    # Multi-direction method options (titan, prism, pulse)
    multi_dir_group = parser.add_argument_group("multi-direction options (titan/prism/pulse)")
    multi_dir_group.add_argument(
        "--num-directions",
        type=int,
        default=5,
        help="Number of directions per layer for multi-direction methods. Default: 5"
    )
    multi_dir_group.add_argument(
        "--combination-strategy",
        type=str,
        default="learned",
        choices=["learned", "uniform", "pca_weighted"],
        help=(
            "How to combine multiple directions when baking into weights. "
            "learned: use method's learned weights, "
            "uniform: equal weights, "
            "pca_weighted: weight by PCA importance. Default: learned"
        )
    )
    multi_dir_group.add_argument(
        "--multi-optimization-steps",
        type=int,
        default=100,
        help="Optimization steps for multi-direction training. Default: 100"
    )

    # ==========================================================================
    # STEERING VECTOR GENERATION
    # ==========================================================================
    vector_group = parser.add_argument_group("steering vector generation")
    vector_group.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Layers for activation collection: 'all' or comma-separated indices. Default: all"
    )
    vector_group.add_argument(
        "--token-aggregation",
        type=str,
        default="average",
        choices=["average", "last", "first", "max"],
        help="Token aggregation strategy. Default: average"
    )
    vector_group.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for synthetic pair filtering. Default: 0.8"
    )
    vector_group.add_argument(
        "--pairs-cache-dir",
        type=str,
        default=None,
        help="Directory to cache/load generated pairs"
    )

    # ==========================================================================
    # HUB EXPORT
    # ==========================================================================
    hub_group = parser.add_argument_group("huggingface hub")
    hub_group.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push optimized model to HuggingFace Hub"
    )
    hub_group.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="HuggingFace repo ID (required if --push-to-hub)"
    )

    # ==========================================================================
    # DISPLAY
    # ==========================================================================
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Show timing information"
    )
    parser.add_argument(
        "--show-comparisons",
        type=int,
        default=0,
        metavar="N",
        help="Show N before/after response comparisons with biggest score changes in console. Default: 0 (disabled)"
    )
    parser.add_argument(
        "--save-comparisons",
        type=str,
        default=None,
        metavar="PATH",
        help="Save all comparisons to JSON file (use with --show-comparisons to also display in console)"
    )
