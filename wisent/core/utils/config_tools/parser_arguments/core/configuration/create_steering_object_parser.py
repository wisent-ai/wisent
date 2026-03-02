"""Parser for the create-steering-object command."""

import argparse

from wisent.core import constants as _C


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
        choices=["caa", "ostrze", "mlp", "tecza", "tetno", "grom", "nurt"],
        required=True,
        help="Steering method to use"
    )

    # Layer selection
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Layer(s) to create steering vectors for. Can be single (e.g., '16'), "
             "comma-separated (e.g., '12,14,16'), or range (e.g., '12-18'). "
             "If not specified, creates vectors for all layers."
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
        default=_C.MLP_HIDDEN_DIM,
        help="Hidden dimension for MLP classifier (default: 256)"
    )
    parser.add_argument(
        "--mlp-num-layers",
        type=int,
        default=_C.MLP_NUM_LAYERS,
        help="Number of hidden layers in MLP (default: 2)"
    )
    parser.add_argument(
        "--mlp-dropout",
        type=float,
        default=_C.MLP_DROPOUT,
        help="Dropout rate for MLP (default: 0.1)"
    )
    parser.add_argument(
        "--mlp-epochs",
        type=int,
        default=_C.VIZ_MLP_EPOCHS,
        help="Training epochs for MLP (default: 100)"
    )
    parser.add_argument(
        "--mlp-learning-rate",
        type=float,
        default=_C.MLP_LEARNING_RATE,
        help="Learning rate for MLP (default: 0.001)"
    )
    parser.add_argument(
        "--mlp-weight-decay",
        type=float,
        default=_C.DEFAULT_WEIGHT_DECAY,
        help="Weight decay for MLP (default: 0.01)"
    )

    # Ostrze parameters
    parser.add_argument(
        "--ostrze-max-iter",
        type=int,
        default=_C.OSTRZE_DEFAULT_ITERATIONS,
        help="Max iterations for logistic regression (default: 1000)"
    )
    parser.add_argument(
        "--ostrze-C",
        type=float,
        default=_C.OSTRZE_C,
        help="Regularization strength for logistic regression (default: 1.0)"
    )

    # TECZA parameters
    parser.add_argument(
        "--tecza-num-directions",
        type=int,
        default=_C.TETNO_DEFAULT_NUM_DIRECTIONS,
        help="Number of directions to discover per layer (default: 3)"
    )
    parser.add_argument(
        "--tecza-optimization-steps",
        type=int,
        default=_C.DEFAULT_OPTIMIZATION_STEPS,
        help="Optimization steps for TECZA (default: 100)"
    )
    parser.add_argument(
        "--tecza-learning-rate",
        type=float,
        default=_C.TETNO_DEFAULT_STEP_SIZE,
        help="Learning rate for TECZA (default: 0.01)"
    )

    # TETNO parameters
    parser.add_argument(
        "--tetno-sensor-layer",
        type=int,
        default=None,
        help="Sensor layer index for gating (default: auto)"
    )
    parser.add_argument(
        "--tetno-condition-threshold",
        type=float,
        default=_C.NURT_DEFAULT_TEMPERATURE,
        help="Condition threshold for gating (default: 0.5)"
    )
    parser.add_argument(
        "--tetno-gate-temperature",
        type=float,
        default=_C.GROM_DEFAULT_LR,
        help="Gate temperature (default: 0.1)"
    )
    parser.add_argument(
        "--tetno-learn-threshold",
        action="store_true",
        default=True,
        help="Learn optimal threshold (default: True)"
    )

    # GROM parameters
    parser.add_argument(
        "--grom-num-directions",
        type=int,
        default=_C.GROM_DEFAULT_NUM_ITER,
        help="Number of directions per layer (default: 5)"
    )
    parser.add_argument(
        "--grom-sensor-layer",
        type=int,
        default=None,
        help="Sensor layer for gating (default: auto)"
    )
    parser.add_argument(
        "--grom-gate-hidden-dim",
        type=int,
        default=None,
        help="Gate network hidden dimension (default: auto)"
    )
    parser.add_argument(
        "--grom-intensity-hidden-dim",
        type=int,
        default=None,
        help="Intensity network hidden dimension (default: auto)"
    )
    parser.add_argument(
        "--grom-max-alpha",
        type=float,
        default=_C.NURT_DEFAULT_STRENGTH,
        help="Maximum steering intensity (default: 3.0)"
    )
    parser.add_argument(
        "--grom-gate-temperature",
        type=float,
        default=_C.GROM_DEFAULT_LAMBDA,
        help="Gate temperature (default: 0.5)"
    )

    # Concept Flow parameters
    parser.add_argument(
        "--nurt-num-dims",
        type=int,
        default=0,
        help="Concept subspace dimensions (0 = auto from variance, default: 0)"
    )
    parser.add_argument(
        "--nurt-variance-threshold",
        type=float,
        default=_C.SZLAK_DEFAULT_EPSILON,
        help="Cumulative variance threshold for auto dim selection (default: 0.80)"
    )
    parser.add_argument(
        "--nurt-training-epochs",
        type=int,
        default=_C.SZLAK_DEFAULT_MAX_ITER,
        help="Training epochs for flow matching (default: 300)"
    )
    parser.add_argument(
        "--nurt-lr",
        type=float,
        default=_C.SZLAK_DEFAULT_STEP_SIZE,
        help="Learning rate for flow network (default: 0.001)"
    )
    parser.add_argument(
        "--nurt-num-integration-steps",
        type=int,
        default=_C.SZLAK_DEFAULT_ITER_PER_SCALE,
        help="Euler integration steps at inference (default: 4)"
    )
    parser.add_argument(
        "--nurt-t-max",
        type=float,
        default=_C.PRZELOM_DEFAULT_INFERENCE_K,
        help="Integration endpoint / max steering strength (default: 1.0)"
    )
    parser.add_argument(
        "--nurt-hidden-dim",
        type=int,
        default=0,
        help="Velocity network hidden dim (0 = auto, default: 0)"
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
