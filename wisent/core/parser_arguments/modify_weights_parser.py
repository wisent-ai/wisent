"""
Parser for modify-weights command.

This command modifies model weights permanently using steering vectors.
"""

import argparse


def setup_modify_weights_parser(parser: argparse.ArgumentParser) -> None:
    """
    Set up argument parser for modify-weights command.

    This command generates steering vectors and permanently modifies model weights
    using either directional projection or additive methods.
    """

    # Input source (mutually exclusive: task or steering-vectors)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--task",
        type=str,
        help=(
            "Task to modify weights for. Can be: "
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
        help="Path to pre-computed steering vectors file (.json or .pt)"
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
    
    # Additional options for multi-benchmark mode
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

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save modified model"
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier (HuggingFace model ID or path)"
    )

    # Contrastive pair generation (for task/trait modes)
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=100,
        help="Number of contrastive pairs to generate (default: 100)"
    )
    parser.add_argument(
        "--trait-label",
        type=str,
        default="correctness",
        help="Trait label for task mode (default: correctness)"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for synthetic mode (default: 0.8)"
    )
    parser.add_argument(
        "--pairs-cache-dir",
        type=str,
        default=None,
        help="Directory to cache/load pairs. Speeds up repeated runs with same trait."
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force regeneration of pairs even if cached pairs exist"
    )

    # Activation collection
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Layers to collect activations from: 'all', single number (e.g. '8'), or comma-separated (e.g. '8,12,15'). Default: all layers"
    )
    parser.add_argument(
        "--token-aggregation",
        type=str,
        default="average",
        choices=["average", "last", "first"],
        help="How to aggregate token activations (default: average)"
    )
    parser.add_argument(
        "--prompt-strategy",
        type=str,
        default="chat_template",
        choices=["chat_template", "raw"],
        help="Prompt formatting strategy (default: chat_template)"
    )

    # Steering vector generation method
    parser.add_argument(
        "--steering-method",
        type=str,
        default="auto",
        choices=["auto", "caa", "hyperplane", "prism", "pulse", "titan", "mlp"],
        help="Method for generating steering vectors: 'auto' (analyze geometry and pick best), 'caa' (mean difference), 'hyperplane' (SVM), 'prism' (sparse decomposition), 'pulse' (optimized), 'titan' (manifold-based), 'mlp' (neural probe). Default: auto"
    )

    # Weight modification method
    modification_group = parser.add_argument_group("weight modification")
    modification_group.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=["auto", "directional", "additive", "titan", "pulse", "prism"],
        help="Weight modification method: 'auto' (analyze geometry and pick best), 'directional' (project onto/away from direction), 'additive' (add bias), 'titan' (manifold with dynamic gating), 'pulse' (conditional gating), or 'prism' (multi-directional). Default: auto"
    )
    
    # TITAN-specific parameters
    titan_group = parser.add_argument_group("titan parameters")
    titan_group.add_argument(
        "--titan-mode",
        type=str,
        default="hybrid",
        choices=["static", "dynamic", "hybrid"],
        help="TITAN mode: 'static' (bake directions only), 'dynamic' (hooks only), 'hybrid' (both, recommended). Default: hybrid"
    )
    titan_group.add_argument(
        "--titan-num-directions",
        type=int,
        default=8,
        help="Number of manifold directions for TITAN (default: 8)"
    )

    # PRISM-specific parameters
    prism_group = parser.add_argument_group("prism parameters")
    prism_group.add_argument(
        "--prism-mode",
        type=str,
        default="weighted",
        choices=["primary", "weighted", "full"],
        help="PRISM mode: 'primary' (only first direction), 'weighted' (average all), 'full' (save all). Default: weighted"
    )
    prism_group.add_argument(
        "--prism-num-directions",
        type=int,
        default=3,
        help="Number of directions per layer for PRISM (default: 3)"
    )

    # Directional projection parameters
    directional_group = parser.add_argument_group("directional projection parameters")
    directional_group.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Projection strength (0=no change, 1=full projection) (default: 1.0)"
    )
    directional_group.add_argument(
        "--norm-preserve",
        action="store_true",
        default=True,
        help="Use norm-preserving projection (RECOMMENDED, default: True)"
    )
    directional_group.add_argument(
        "--no-norm-preserve",
        action="store_true",
        help="Disable norm-preserving projection (NOT recommended)"
    )
    directional_group.add_argument(
        "--use-biprojection",
        action="store_true",
        default=True,
        help="Orthogonalize steering direction against harmless direction (default: True)"
    )
    directional_group.add_argument(
        "--no-biprojection",
        action="store_true",
        help="Disable biprojection (NOT recommended)"
    )
    directional_group.add_argument(
        "--harmless-vectors",
        type=str,
        default=None,
        help="Path to harmless direction vectors JSON for biprojection (optional)"
    )

    # Additive-specific parameters
    additive_group = parser.add_argument_group("additive parameters")
    additive_group.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Steering strength for additive method (default: 1.0)"
    )
    additive_group.add_argument(
        "--additive-method",
        type=str,
        default="bias",
        choices=["bias", "weight"],
        help="How to add steering: 'bias' (add to output bias) or 'weight' (modify weight matrix) (default: bias)"
    )

    # Component selection
    parser.add_argument(
        "--components",
        type=str,
        nargs="+",
        default=None,
        help="Components to modify (e.g., 'self_attn.o_proj' 'mlp.down_proj'). Default: method-specific defaults"
    )

    # Kernel-based layer weighting
    kernel_group = parser.add_argument_group("kernel-based layer weighting")
    kernel_group.add_argument(
        "--use-kernel",
        action="store_true",
        help="Use Gaussian-like kernel for smooth layer weighting"
    )
    kernel_group.add_argument(
        "--max-weight",
        type=float,
        default=1.5,
        help="Peak weight/alpha at center layer (default: 1.5)"
    )
    kernel_group.add_argument(
        "--max-weight-position",
        type=float,
        default=None,
        help="Layer index for peak weight (default: middle layer)"
    )
    kernel_group.add_argument(
        "--min-weight",
        type=float,
        default=0.3,
        help="Minimum weight/alpha at edges (default: 0.3)"
    )
    kernel_group.add_argument(
        "--min-weight-distance",
        type=float,
        default=None,
        help="Distance over which weight decays (default: 60%% of layers)"
    )

    # Vector processing
    parser.add_argument(
        "--normalize-vectors",
        action="store_true",
        help="Normalize steering vectors before modification"
    )

    # Export options
    export_group = parser.add_argument_group("export options")
    export_group.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload modified model to HuggingFace Hub"
    )
    export_group.add_argument(
        "--repo-id",
        type=str,
        help="HuggingFace repository ID (required if --push-to-hub)"
    )
    export_group.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Commit message for Hub upload"
    )
    export_group.add_argument(
        "--private",
        action="store_true",
        help="Make Hub repository private"
    )

    # Optimal config usage
    parser.add_argument(
        "--no-optimal",
        action="store_false",
        dest="use_optimal",
        default=True,
        help="Don't use optimal config from previous optimization (use defaults instead)"
    )

    # Guided modification (linearity-driven) - NOVEL FEATURES
    guided_group = parser.add_argument_group("guided modification (linearity-driven)")
    guided_group.add_argument(
        "--guided",
        action="store_true",
        help="Use linearity-guided weight modification. Automatically selects layers and weights based on measured linear separability."
    )
    guided_group.add_argument(
        "--guided-mode",
        type=str,
        default="adaptive",
        choices=["full", "surgical", "adaptive"],
        help=(
            "Guided ablation mode: "
            "'full' (all layers with signal), "
            "'surgical' (only top-k layers), "
            "'adaptive' (auto-select based on variance). Default: adaptive"
        )
    )
    guided_group.add_argument(
        "--surgical-top-k",
        type=int,
        default=3,
        help="Number of top layers for surgical mode (default: 3)"
    )
    guided_group.add_argument(
        "--min-linear-score",
        type=float,
        default=0.5,
        help="Minimum linear score to include a layer (default: 0.5)"
    )
    guided_group.add_argument(
        "--use-fisher-weights",
        action="store_true",
        default=True,
        help="Weight ablation strength by Fisher ratio (default: True)"
    )
    guided_group.add_argument(
        "--no-fisher-weights",
        action="store_true",
        help="Disable Fisher ratio weighting"
    )
    guided_group.add_argument(
        "--extraction-strategy",
        type=str,
        default="chat_last",
        choices=["chat_last", "chat_mean", "chat_max_norm", "completion_last"],
        help="Extraction strategy for guided mode (default: chat_last)"
    )
    
    # Collateral damage validation
    validation_group = parser.add_argument_group("collateral damage validation")
    validation_group.add_argument(
        "--validate-collateral",
        action="store_true",
        help="Validate that modification doesn't hurt unrelated representations"
    )
    validation_group.add_argument(
        "--max-degradation",
        type=float,
        default=0.1,
        help="Maximum allowed degradation on validation benchmarks (default: 0.1)"
    )
    validation_group.add_argument(
        "--validation-benchmarks",
        type=str,
        nargs="+",
        default=None,
        help="Benchmarks to use for collateral damage validation"
    )
    
    # Multi-concept modification
    multi_concept_group = parser.add_argument_group("multi-concept modification")
    multi_concept_group.add_argument(
        "--concepts",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Multiple concepts to modify simultaneously. Format: 'name:action:strength' "
            "e.g., 'refusal:suppress:1.0' 'truthfulness:enhance:0.5'. "
            "Actions: suppress, enhance"
        )
    )
    multi_concept_group.add_argument(
        "--orthogonalize-concepts",
        action="store_true",
        default=True,
        help="Orthogonalize concept directions to minimize interference (default: True)"
    )
    multi_concept_group.add_argument(
        "--no-orthogonalize",
        action="store_true",
        help="Disable concept orthogonalization"
    )

    # General options
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
        "--save-steering-vectors",
        type=str,
        default=None,
        help="Save generated steering vectors to file (optional)"
    )
    parser.add_argument(
        "--save-diagnostics",
        type=str,
        default=None,
        help="Save layer diagnostics to JSON file (for guided mode)"
    )
