"""Parser setup for the 'agent' command."""
from wisent.core.utils.config_tools.constants import (
    AGENT_CLASSIFIER_EPOCHS,
    AGENT_DECISION_QUALITY_THRESHOLD,
    AGENT_DECISION_TIME_BUDGET,
    AGENT_MAX_PARALLEL_WORKERS,
    AGENT_MAX_RESPONSE_ATTEMPTS,
    DEFAULT_CLASSIFIER_LR,
    QUALITY_CONTROL_MAX_ATTEMPTS,
)


def setup_agent_parser(parser):
    """Set up the agent subcommand parser."""
    parser.add_argument("prompt", type=str, help="Prompt to send to the autonomous agent")
    parser.add_argument("--model", type=str, required=True, help="Model to use")
    parser.add_argument("--layer", type=int, help="Layer to use (overrides parameter file)")
    parser.add_argument(
        "--agent-strategy",
        type=str,
        required=True,
        choices=["synthetic_pairs_classifier_steering"],
        help="Agent strategy to use"
    )
    parser.add_argument(
        "--quality-threshold", type=float, default=AGENT_DECISION_QUALITY_THRESHOLD, help="Quality threshold for classifiers (default: 0.3)"
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=AGENT_DECISION_TIME_BUDGET,
        help="Time budget in minutes for creating classifiers (default: 10.0)",
    )
    parser.add_argument("--max-attempts", type=int, default=AGENT_MAX_RESPONSE_ATTEMPTS, help="Maximum improvement attempts (default: 3)")
    parser.add_argument(
        "--max-classifiers", type=int, default=None, help="Maximum classifiers to use (default: no limit)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # Synthetic pair generation arguments
    parser.add_argument(
        "--num-pairs", type=int, default=None, help="Number of pairs to generate (default: based on time budget)"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Similarity threshold for deduplication (0-1, higher = more strict)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=AGENT_MAX_PARALLEL_WORKERS, help="Number of parallel workers for generation (default: 4)"
    )

    # Activation collection arguments
    parser.add_argument(
        "--normalize-layers", action="store_true", help="Normalize layer activations"
    )
    parser.add_argument(
        "--return-full-sequence", action="store_true", help="Return full activation sequence instead of aggregated"
    )

    # Classifier training arguments
    parser.add_argument(
        "--classifier-epochs", type=int, default=AGENT_CLASSIFIER_EPOCHS, help="Number of epochs for classifier training (default: 50)"
    )
    parser.add_argument(
        "--classifier-lr", type=float, default=DEFAULT_CLASSIFIER_LR, help="Learning rate for classifier training (default: 1e-3)"
    )
    parser.add_argument(
        "--classifier-batch-size",
        type=int,
        default=None,
        help="Batch size for classifier training (default: adaptive based on data size)",
    )
    parser.add_argument(
        "--classifier-type",
        type=str,
        required=True,
        choices=["logistic", "mlp"],
        help="Type of classifier to use",
    )

    # Steering method arguments - uses centralized registry
    from wisent.core.control.steering_methods import SteeringMethodRegistry
    SteeringMethodRegistry.add_all_cli_arguments(parser)
    parser.add_argument(
        "--steering-strength", type=float, required=True, help="Strength of steering vector application"
    )
    parser.add_argument("--steering-mode", action="store_true", help="Enable steering mode")

    # Normalization parameters
    parser.add_argument("--normalize-mode", action="store_true", help="Enable normalization of steering vectors")
    parser.add_argument(
        "--normalization-method",
        type=str,
        required=True,
        choices=["none", "l2_unit", "l2_norm", "max_norm"],
        help="Normalization method for steering vectors",
    )
    parser.add_argument("--target-norm", type=float, default=None, help="Target norm for steering vectors")

    # Quality Control System parameters
    parser.add_argument(
        "--enable-quality-control",
        action="store_true",
        default=True,
        help="Enable new quality control system (default: True)",
    )
    parser.add_argument(
        "--max-quality-attempts",
        type=int,
        default=QUALITY_CONTROL_MAX_ATTEMPTS,
        help="Maximum attempts to achieve acceptable quality (default: 5)",
    )
    parser.add_argument(
        "--show-parameter-reasoning", action="store_true", help="Display model's reasoning for parameter choices"
    )
