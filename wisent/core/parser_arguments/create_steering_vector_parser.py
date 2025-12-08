"""Parser for the create-steering-vector command."""

import argparse


def setup_create_steering_vector_parser(parser: argparse.ArgumentParser) -> None:
    """
    Set up the create-steering-vector command parser.

    This command loads enriched pairs (with activations) from JSON and creates
    steering vectors using a specified method (e.g., CAA).
    """
    # Input/Output
    parser.add_argument(
        "enriched_pairs_file",
        type=str,
        help="Path to JSON file containing contrastive pairs with activations"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path for steering vector (JSON)"
    )

    # Steering method
    parser.add_argument(
        "--method",
        type=str,
        choices=["caa"],
        default="caa",
        help="Steering method to use (default: caa)"
    )

    # Method parameters
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="L2-normalize steering vectors (default: True)"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_false",
        dest="normalize",
        help="Do not L2-normalize steering vectors"
    )

    # Quality control
    parser.add_argument(
        "--accept-low-quality-vector",
        action="store_true",
        default=False,
        help="Accept steering vectors that fail quality checks (convergence, SNR, etc.)"
    )

    # Display options
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
