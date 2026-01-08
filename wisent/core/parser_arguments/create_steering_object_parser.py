"""Parser for the create-steering-object command."""

import argparse


def setup_create_steering_object_parser(parser: argparse.ArgumentParser) -> None:
    """
    Set up the create-steering-object command parser.

    This command creates full steering objects that preserve method-specific
    components like gates, intensity networks, multi-directions, etc.
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
        help="Output file path for steering object (.pt or .json)"
    )

    # Steering method
    parser.add_argument(
        "--method",
        type=str,
        choices=["caa", "hyperplane", "mlp", "prism", "pulse", "titan"],
        default="caa",
        help="Steering method to use (default: caa)"
    )

    # Common parameters
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

    # MLP parameters
    parser.add_argument(
        "--mlp-hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension for MLP classifier (default: 256)"
    )
    parser.add_argument(
        "--mlp-num-layers",
        type=int,
        default=2,
        help="Number of hidden layers in MLP (default: 2)"
    )
    parser.add_argument(
        "--mlp-dropout",
        type=float,
        default=0.1,
        help="Dropout rate for MLP (default: 0.1)"
    )
    parser.add_argument(
        "--mlp-epochs",
        type=int,
        default=100,
        help="Training epochs for MLP (default: 100)"
    )
    parser.add_argument(
        "--mlp-learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for MLP (default: 0.001)"
    )
    parser.add_argument(
        "--mlp-weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for MLP (default: 0.01)"
    )

    # Hyperplane parameters
    parser.add_argument(
        "--hyperplane-max-iter",
        type=int,
        default=1000,
        help="Max iterations for logistic regression (default: 1000)"
    )
    parser.add_argument(
        "--hyperplane-C",
        type=float,
        default=1.0,
        help="Regularization strength for logistic regression (default: 1.0)"
    )

    # PRISM parameters
    parser.add_argument(
        "--prism-num-directions",
        type=int,
        default=3,
        help="Number of directions to discover per layer (default: 3)"
    )
    parser.add_argument(
        "--prism-optimization-steps",
        type=int,
        default=100,
        help="Optimization steps for PRISM (default: 100)"
    )
    parser.add_argument(
        "--prism-learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for PRISM (default: 0.01)"
    )

    # PULSE parameters
    parser.add_argument(
        "--pulse-sensor-layer",
        type=int,
        default=None,
        help="Sensor layer index for gating (default: auto)"
    )
    parser.add_argument(
        "--pulse-condition-threshold",
        type=float,
        default=0.5,
        help="Condition threshold for gating (default: 0.5)"
    )
    parser.add_argument(
        "--pulse-gate-temperature",
        type=float,
        default=0.1,
        help="Gate temperature (default: 0.1)"
    )
    parser.add_argument(
        "--pulse-learn-threshold",
        action="store_true",
        default=True,
        help="Learn optimal threshold (default: True)"
    )

    # TITAN parameters
    parser.add_argument(
        "--titan-num-directions",
        type=int,
        default=5,
        help="Number of directions per layer (default: 5)"
    )
    parser.add_argument(
        "--titan-sensor-layer",
        type=int,
        default=None,
        help="Sensor layer for gating (default: auto)"
    )
    parser.add_argument(
        "--titan-gate-hidden-dim",
        type=int,
        default=None,
        help="Gate network hidden dimension (default: auto)"
    )
    parser.add_argument(
        "--titan-intensity-hidden-dim",
        type=int,
        default=None,
        help="Intensity network hidden dimension (default: auto)"
    )
    parser.add_argument(
        "--titan-max-alpha",
        type=float,
        default=3.0,
        help="Maximum steering intensity (default: 3.0)"
    )
    parser.add_argument(
        "--titan-gate-temperature",
        type=float,
        default=0.5,
        help="Gate temperature (default: 0.5)"
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
