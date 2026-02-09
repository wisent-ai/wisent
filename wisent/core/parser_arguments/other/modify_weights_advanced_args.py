"""Advanced argument groups for modify-weights command.

Guided modification, collateral damage validation, multi-concept,
and general options.
"""

import argparse


def add_guided_args(parser: argparse.ArgumentParser) -> None:
    """Add guided modification (linearity-driven) arguments."""
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


def add_validation_args(parser: argparse.ArgumentParser) -> None:
    """Add collateral damage validation arguments."""
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


def add_multi_concept_args(parser: argparse.ArgumentParser) -> None:
    """Add multi-concept modification arguments."""
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
    multi_concept_group.add_argument(
        "--use-null-space",
        action="store_true",
        help="Use AlphaEdit-style null-space projection to prevent interference with preserved activations"
    )
    multi_concept_group.add_argument(
        "--null-space-epsilon",
        type=float,
        default=1e-6,
        help="Tikhonov regularization for null-space projector SVD (default: 1e-6)"
    )
    multi_concept_group.add_argument(
        "--null-space-max-rank",
        type=int,
        default=None,
        help="Optional SVD rank truncation for null-space projector"
    )


def add_general_args(parser: argparse.ArgumentParser) -> None:
    """Add general options."""
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
