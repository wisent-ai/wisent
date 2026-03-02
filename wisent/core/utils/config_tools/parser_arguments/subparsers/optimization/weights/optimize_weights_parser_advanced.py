"""Advanced arguments for optimize-weights parser."""
import argparse

from wisent.core.utils.config_tools.constants import (
    DEFAULT_OPTIMIZATION_STEPS,
    DIRECTIONS_PER_LAYER,
    PAIRS_SIMILARITY_THRESHOLD,
    WEIGHT_MOD_DEFAULT_COMPONENTS,
    WEIGHT_OPT_STRENGTH_RANGE,
    WEIGHT_OPT_MAX_WEIGHT_RANGE,
    WEIGHT_OPT_MIN_WEIGHT_RANGE,
    WEIGHT_OPT_POSITION_RANGE,
)


def setup_advanced_optimize_weights_args(parser: argparse.ArgumentParser) -> None:
    """Set up advanced optimize-weights arguments."""

    # ==========================================================================
    # OPTIMIZATION PARAMETERS
    # ==========================================================================
    optim_group = parser.add_argument_group("optimization")
    optim_group.add_argument(
        "--trials",
        type=int,
        required=True,
        help="Number of optimization trials"
    )
    optim_group.add_argument(
        "--startup-trials",
        type=int,
        required=True,
        help="Number of random startup trials before TPE"
    )
    optim_group.add_argument(
        "--early-stop",
        action="store_true",
        help="Stop early if target value is reached"
    )
    optim_group.add_argument(
        "--early-stop-patience",
        type=int,
        required=True,
        help="Stop if no improvement for N trials"
    )

    # ==========================================================================
    # SEARCH SPACE: Parameters to optimize
    # ==========================================================================
    search_group = parser.add_argument_group("search space")
    search_group.add_argument(
        "--strength-range",
        type=str,
        default=WEIGHT_OPT_STRENGTH_RANGE,
        help="Min,max range for directional projection strength. Default: 0.3,2.0"
    )
    search_group.add_argument(
        "--max-weight-range",
        type=str,
        default=WEIGHT_OPT_MAX_WEIGHT_RANGE,
        help="Min,max range for kernel max weight. Default: 0.5,3.0"
    )
    search_group.add_argument(
        "--min-weight-range",
        type=str,
        default=WEIGHT_OPT_MIN_WEIGHT_RANGE,
        help="Min,max range for kernel min weight. Default: 0.0,0.5"
    )
    search_group.add_argument(
        "--position-range",
        type=str,
        default=WEIGHT_OPT_POSITION_RANGE,
        help="Min,max range for kernel peak position (as ratio 0-1). Default: 0.3,0.7"
    )
    search_group.add_argument(
        "--num-pairs",
        type=int,
        required=True,
        help="Number of contrastive pairs to generate (fixed, not optimized)"
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
        required=True,
        choices=["directional", "additive", "grom", "tecza", "tetno"],
        help=(
            "Weight modification method. "
            "Options: directional (single direction), additive (bias-based), "
            "grom/tecza/tetno (multi-direction, better for non-linear representations)"
        )
    )
    method_group.add_argument(
        "--components",
        type=str,
        nargs="+",
        default=list(WEIGHT_MOD_DEFAULT_COMPONENTS),
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
    
    # Multi-direction method options (grom, tecza, tetno)
    multi_dir_group = parser.add_argument_group("multi-direction options (grom/tecza/tetno)")
    multi_dir_group.add_argument(
        "--num-directions",
        type=int,
        default=DIRECTIONS_PER_LAYER,
        help="Number of directions per layer for multi-direction methods. Default: 5"
    )
    multi_dir_group.add_argument(
        "--combination-strategy",
        type=str,
        required=True,
        choices=["learned", "uniform", "pca_weighted"],
        help=(
            "How to combine multiple directions when baking into weights. "
            "learned: use method's learned weights, "
            "uniform: equal weights, "
            "pca_weighted: weight by PCA importance"
        )
    )
    multi_dir_group.add_argument(
        "--multi-optimization-steps",
        type=int,
        default=DEFAULT_OPTIMIZATION_STEPS,
        help="Optimization steps for multi-direction training. Default: 100"
    )

    # ==========================================================================
    # STEERING VECTOR GENERATION
    # ==========================================================================
    vector_group = parser.add_argument_group("steering vector generation")
    vector_group.add_argument(
        "--layers",
        type=str,
        required=True,
        help="Layers for activation collection: 'all' or comma-separated indices"
    )
    vector_group.add_argument(
        "--similarity-threshold",
        type=float,
        default=PAIRS_SIMILARITY_THRESHOLD,
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
