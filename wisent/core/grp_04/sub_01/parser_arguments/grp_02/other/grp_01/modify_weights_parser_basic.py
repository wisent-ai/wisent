"""Basic arguments for modify-weights parser."""
import argparse


def setup_basic_modify_args(parser: argparse.ArgumentParser) -> None:
    """Set up basic modification arguments."""
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
    # Steering vector generation method
    parser.add_argument(
        "--steering-method",
        type=str,
        default="auto",
        choices=["auto", "caa", "ostrze", "tecza", "tetno", "grom", "mlp", "szlak", "wicher", "nurt"],
        help="Method for generating steering vectors: auto, caa, ostrze, tecza, tetno, grom, mlp, szlak (geodesic OT), wicher (Newton), nurt (concept flow). Default: auto"
    )

    # Weight modification method
    modification_group = parser.add_argument_group("weight modification")
    modification_group.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=["auto", "directional", "additive", "grom", "tetno", "tecza", "nurt", "szlak", "wicher"],
        help="Weight modification method: auto, directional, additive, grom, tetno, tecza, nurt (concept flow), szlak (geodesic OT), wicher (Newton). Default: auto"
    )
    
    # GROM-specific parameters
    grom_group = parser.add_argument_group("grom parameters")
    grom_group.add_argument(
        "--grom-mode",
        type=str,
        default="hybrid",
        choices=["static", "dynamic", "hybrid"],
        help="GROM mode: 'static' (bake directions only), 'dynamic' (hooks only), 'hybrid' (both, recommended). Default: hybrid"
    )
    grom_group.add_argument(
        "--grom-num-directions",
        type=int,
        default=8,
        help="Number of manifold directions for GROM (default: 8)"
    )

    # TECZA-specific parameters
    tecza_group = parser.add_argument_group("tecza parameters")
    tecza_group.add_argument(
        "--tecza-mode",
        type=str,
        default="weighted",
        choices=["primary", "weighted", "full"],
        help="TECZA mode: 'primary' (only first direction), 'weighted' (average all), 'full' (save all). Default: weighted"
    )
    tecza_group.add_argument(
        "--tecza-num-directions",
        type=int,
        default=3,
        help="Number of directions per layer for TECZA (default: 3)"
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