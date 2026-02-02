"""Parser for verify-steering command."""

import argparse


def setup_verify_steering_parser(parser: argparse.ArgumentParser) -> None:
    """Set up arguments for the verify-steering command.

    This command verifies that a steered model's activations are correctly
    aligned with the intended steering direction at inference time.
    """
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the steered model (TITAN, PULSE, or CAA)"
    )

    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Path or name of base model for comparison (auto-detected from config if not specified)"
    )

    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=None,
        help="Test prompts to verify steering on (defaults to built-in test prompts)"
    )

    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="JSON file containing test prompts (list of strings or objects with 'prompt' key)"
    )

    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices to check (default: all steering layers)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for inference (default: auto)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for detailed results (JSON format)"
    )

    parser.add_argument(
        "--alignment-threshold",
        type=float,
        default=0.3,
        help="Minimum alignment score to consider steering successful (default: 0.3)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-layer diagnostics"
    )

    parser.add_argument(
        "--check-gate",
        action="store_true",
        default=True,
        help="Check gate network discrimination (for TITAN/PULSE models)"
    )

    parser.add_argument(
        "--check-intensity",
        action="store_true",
        default=True,
        help="Check intensity network predictions (for TITAN models)"
    )
