"""Parser for modify-weights command."""

import argparse

from .modify_weights_advanced_args import (
    add_guided_args, add_validation_args,
    add_multi_concept_args, add_general_args,
)


def setup_modify_weights_parser(parser: argparse.ArgumentParser) -> None:
    """Set up argument parser for modify-weights command."""

    # Input source (mutually exclusive: task or steering-vectors)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--task", type=str,
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
        "--steering-vectors", type=str,
        help="Path to pre-computed steering vectors file (.json or .pt)"
    )

    parser.add_argument("--trait", type=str, default=None,
                        help="Trait description for personalization (required when --task personalization)")
    parser.add_argument(
        "--custom-evaluator", type=str, default=None,
        help="Custom evaluator specification (required when --task custom). Can be: "
             "(1) Python module path, (2) File path with function, (3) Built-in example"
    )
    parser.add_argument("--custom-evaluator-kwargs", type=str, default=None,
                        help='JSON string of kwargs for custom evaluator')
    parser.add_argument("--cap-pairs-per-benchmark", type=int, default=None,
                        help="Cap pairs per benchmark. Benchmarks with more pairs get randomly sampled.")
    parser.add_argument("--max-benchmarks", type=int, default=None,
                        help="Maximum number of benchmarks to use")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Fraction of pairs for training vs evaluation (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save modified model")
    parser.add_argument("--model", type=str, required=True, help="Model identifier (HuggingFace model ID or path)")
    parser.add_argument("--num-pairs", type=int, default=100, help="Number of contrastive pairs (default: 100)")
    parser.add_argument("--trait-label", type=str, default="correctness",
                        help="Trait label for task mode (default: correctness)")
    parser.add_argument("--similarity-threshold", type=float, default=0.8,
                        help="Similarity threshold for synthetic mode (default: 0.8)")
    parser.add_argument("--pairs-cache-dir", type=str, default=None,
                        help="Directory to cache/load pairs")
    parser.add_argument("--force-regenerate", action="store_true",
                        help="Force regeneration of pairs even if cached pairs exist")
    parser.add_argument(
        "--layers", type=str, default=None,
        help="Layers to collect activations from: 'all', single number, or comma-separated. Default: all"
    )
    parser.add_argument(
        "--steering-method", type=str, default="auto",
        choices=["auto", "caa", "hyperplane", "prism", "pulse", "titan", "mlp"],
        help="Method for generating steering vectors (default: auto)"
    )

    # Weight modification method
    modification_group = parser.add_argument_group("weight modification")
    modification_group.add_argument(
        "--method", type=str, default="auto",
        choices=["auto", "directional", "additive", "titan", "pulse", "prism"],
        help="Weight modification method (default: auto)"
    )

    # TITAN parameters
    titan_group = parser.add_argument_group("titan parameters")
    titan_group.add_argument("--titan-mode", type=str, default="hybrid",
                             choices=["static", "dynamic", "hybrid"], help="TITAN mode (default: hybrid)")
    titan_group.add_argument("--titan-num-directions", type=int, default=8,
                             help="Number of manifold directions for TITAN (default: 8)")

    # PRISM parameters
    prism_group = parser.add_argument_group("prism parameters")
    prism_group.add_argument("--prism-mode", type=str, default="weighted",
                             choices=["primary", "weighted", "full"], help="PRISM mode (default: weighted)")
    prism_group.add_argument("--prism-num-directions", type=int, default=3,
                             help="Directions per layer for PRISM (default: 3)")

    # Directional projection parameters
    directional_group = parser.add_argument_group("directional projection parameters")
    directional_group.add_argument("--strength", type=float, default=1.0,
                                   help="Projection strength (default: 1.0)")
    directional_group.add_argument("--norm-preserve", action="store_true", default=True,
                                   help="Use norm-preserving projection (RECOMMENDED, default: True)")
    directional_group.add_argument("--no-norm-preserve", action="store_true",
                                   help="Disable norm-preserving projection (NOT recommended)")
    directional_group.add_argument("--use-biprojection", action="store_true", default=True,
                                   help="Orthogonalize against harmless direction (default: True)")
    directional_group.add_argument("--no-biprojection", action="store_true",
                                   help="Disable biprojection (NOT recommended)")
    directional_group.add_argument("--harmless-vectors", type=str, default=None,
                                   help="Path to harmless direction vectors JSON for biprojection")

    # Additive parameters
    additive_group = parser.add_argument_group("additive parameters")
    additive_group.add_argument("--alpha", type=float, default=1.0,
                                help="Steering strength for additive method (default: 1.0)")
    additive_group.add_argument("--additive-method", type=str, default="bias",
                                choices=["bias", "weight"], help="How to add steering (default: bias)")

    parser.add_argument("--components", type=str, nargs="+", default=None,
                        help="Components to modify (e.g., 'self_attn.o_proj' 'mlp.down_proj')")

    # Kernel-based layer weighting
    kernel_group = parser.add_argument_group("kernel-based layer weighting")
    kernel_group.add_argument("--use-kernel", action="store_true", help="Use Gaussian-like kernel for layer weighting")
    kernel_group.add_argument("--max-weight", type=float, default=1.5, help="Peak weight at center (default: 1.5)")
    kernel_group.add_argument("--max-weight-position", type=float, default=None, help="Layer index for peak weight")
    kernel_group.add_argument("--min-weight", type=float, default=0.3, help="Min weight at edges (default: 0.3)")
    kernel_group.add_argument("--min-weight-distance", type=float, default=None, help="Decay distance")

    parser.add_argument("--normalize-vectors", action="store_true", help="Normalize steering vectors before modification")

    # Export options
    export_group = parser.add_argument_group("export options")
    export_group.add_argument("--push-to-hub", action="store_true", help="Upload modified model to HuggingFace Hub")
    export_group.add_argument("--repo-id", type=str, help="HuggingFace repository ID (required if --push-to-hub)")
    export_group.add_argument("--commit-message", type=str, default=None, help="Commit message for Hub upload")
    export_group.add_argument("--private", action="store_true", help="Make Hub repository private")

    parser.add_argument("--no-optimal", action="store_false", dest="use_optimal", default=True,
                        help="Don't use optimal config from previous optimization")

    # Advanced argument groups (guided, validation, multi-concept, general)
    add_guided_args(parser)
    add_validation_args(parser)
    add_multi_concept_args(parser)
    add_general_args(parser)
