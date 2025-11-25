"""
Parser for modify-weights command.

This command modifies model weights permanently using steering vectors.
"""

import argparse


def setup_modify_weights_parser(parser: argparse.ArgumentParser) -> None:
    """
    Set up argument parser for modify-weights command.

    This command generates steering vectors and permanently modifies model weights
    using either abliteration (orthogonal projection) or additive methods.
    """

    # Input source (mutually exclusive: task or trait)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--task",
        type=str,
        help="LM-eval task to generate contrastive pairs from"
    )
    input_group.add_argument(
        "--trait",
        type=str,
        help="Trait for synthetic contrastive pair generation"
    )
    input_group.add_argument(
        "--steering-vectors",
        type=str,
        help="Path to pre-computed steering vectors JSON file"
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

    # Weight modification method
    modification_group = parser.add_argument_group("weight modification")
    modification_group.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["abliteration", "additive"],
        help="Weight modification method: 'abliteration' (remove capability) or 'additive' (enhance behavior)"
    )

    # Abliteration-specific parameters
    abliteration_group = parser.add_argument_group("abliteration parameters")
    abliteration_group.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Abliteration strength (0=no change, 1=full abliteration) (default: 1.0)"
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
