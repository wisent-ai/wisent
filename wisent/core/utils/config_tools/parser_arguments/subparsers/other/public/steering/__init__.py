"""Steering-related parser arguments."""

from .multi_steer_parser import setup_multi_steer_parser
from .steering_viz_parser import setup_steering_viz_parser
from .discover_steering_parser import setup_discover_steering_parser


def setup_find_best_method_parser(parser):
    """Add arguments for the find-best-method command."""
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--task", required=True,
        help="Benchmark task name",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory for results",
    )
    parser.add_argument(
        "--trials-multiplier", type=int, required=True,
        help="Trials per dimension",
    )
    parser.add_argument(
        "--backend", required=True,
        choices=["hyperopt", "optuna"],
        help="Optimizer backend",
    )


__all__ = [
    'setup_multi_steer_parser',
    'setup_steering_viz_parser',
    'setup_discover_steering_parser',
    'setup_find_best_method_parser',
]
