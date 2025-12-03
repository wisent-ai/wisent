"""
Parser for optimize-weights command.

This command runs an optimization loop to find optimal weight modification
parameters for any trait or task, using any evaluator.

Unified pipeline:
1. Generate steering vector (from trait or task)
2. Apply weight modification with trial parameters
3. Evaluate using chosen evaluator
4. Optuna adjusts parameters
5. Repeat until target reached or max trials
6. Save optimized model
"""

import argparse


def setup_optimize_weights_parser(parser: argparse.ArgumentParser) -> None:
    """
    Set up argument parser for optimize-weights command.

    This command optimizes weight modification parameters to achieve a target
    metric value. Works with any trait (synthetic pairs) or task (lm-eval),
    and any evaluator (refusal, task accuracy, personality, custom).
    """

    # ==========================================================================
    # INPUT SOURCE: What to optimize for (trait OR task)
    # ==========================================================================
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--trait",
        type=str,
        help="Trait description for synthetic pair generation (e.g., 'refusal behavior', 'sycophantic responses')"
    )
    input_group.add_argument(
        "--task",
        type=str,
        help="LM-eval task name for task-based optimization (e.g., 'hellaswag', 'arc_easy')"
    )
    input_group.add_argument(
        "--steering-vectors",
        type=str,
        help="Path to pre-computed steering vectors JSON file (skip vector generation)"
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

    # ==========================================================================
    # EVALUATOR CONFIGURATION
    # ==========================================================================
    eval_group = parser.add_argument_group("evaluation")
    eval_group.add_argument(
        "--evaluator",
        type=str,
        default="auto",
        choices=["auto", "refusal", "task", "semantic", "keyword", "llm_judge"],
        help="Evaluator to use. 'auto' selects based on trait/task. Default: auto"
    )
    eval_group.add_argument(
        "--eval-prompts",
        type=str,
        default=None,
        help="Path to custom evaluation prompts JSON file (for refusal/semantic evaluation)"
    )
    eval_group.add_argument(
        "--eval-topics",
        type=str,
        default=None,
        help="Comma-separated UncensorBench topics (for refusal evaluation)"
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
        help="Metric to optimize: compliance_rate, refusal_rate, accuracy, kl_divergence. Default: compliance_rate"
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
    target_group.add_argument(
        "--quality-weight",
        type=float,
        default=0.3,
        help="Weight for quality preservation (KL divergence) in multi-objective. Default: 0.3"
    )

    # ==========================================================================
    # OPTIMIZATION PARAMETERS
    # ==========================================================================
    optim_group = parser.add_argument_group("optimization")
    optim_group.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of optimization trials. Default: 50"
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
        help="Min,max range for abliteration strength. Default: 0.3,2.0"
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
        "--num-pairs-range",
        type=str,
        default="30,100",
        help="Min,max range for number of contrastive pairs. Default: 30,100"
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
        default="abliteration",
        choices=["abliteration", "additive"],
        help="Weight modification method. Default: abliteration"
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
        help="Use norm-preserving abliteration. Default: True"
    )
    method_group.add_argument(
        "--no-norm-preserve",
        action="store_false",
        dest="norm_preserve",
        help="Disable norm-preserving abliteration"
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
