"""Parser setup for the 'optimize-steering' command."""

from wisent.core.control.steering_methods.registry import SteeringMethodRegistry

# Get available steering methods from registry
AVAILABLE_METHODS = [m.upper() for m in SteeringMethodRegistry.list_methods()]



from wisent.core.utils.config_tools.parser_arguments.optimization.steering.optimize_steering_parser_comprehensive import (
    setup_comprehensive_parser,
)
from wisent.core.utils.config_tools.parser_arguments.optimization.steering.optimize_steering_parser_methods import (
    setup_method_parsers,
)
from wisent.core.utils.config_tools.parser_arguments.optimization.steering.optimize_steering_parser_traits import (
    setup_personalization_parsers,
)
from wisent.core.utils.config_tools.constants import (
    CONTINUAL_CONVERGENCE_WINDOW,
    CONTINUAL_EWC_LAMBDA,
    CONTINUAL_FORGETTING_THRESHOLD,
    CONTINUAL_REPLAY_INTERVAL,
    SZLAK_INFERENCE_K,
    TIKHONOV_REG,
)
from wisent.core.utils.config_tools.parser_arguments.optimization.steering.optimize_steering_parser_welfare import (
    setup_welfare_universal_parsers,
)


def _setup_transport_rl_parser(steering_subparsers):
    """Set up the transport-rl optimization subparser."""
    p = steering_subparsers.add_parser(
        "transport-rl",
        help="RL-based steering optimization: cost-matrix shaping for transport "
             "methods (PRZELOM/SZLAK), evolutionary strategy for all others",
    )
    p.add_argument("model", type=str, help="Model name or path")
    p.add_argument("--task", type=str, required=True, help="Benchmark task for evaluation")
    p.add_argument("--enriched-pairs-file", type=str, required=True,
                   dest="enriched_pairs_file", help="Enriched pairs JSON with Q/K projections")
    p.add_argument("--method", type=str, default="przelom",
                   help="Steering method (default: przelom). Transport methods "
                        "(przelom, szlak) use cost-matrix REINFORCE; all others use vector ES")
    p.add_argument("--max-iterations", type=int, required=True, dest="max_iterations",
                   help="RL iteration count")
    p.add_argument("--learning-rate", type=float, required=True, dest="learning_rate",
                   help="Step size for cost shaping or ES update")
    p.add_argument("--epsilon", type=float, required=True,
                   help="Entropic regularization temperature for transport methods")
    p.add_argument("--regularization", type=float, default=TIKHONOV_REG,
                   help="Tikhonov regularization for pseudoinverse (default: 1e-4)")
    p.add_argument("--inference-k", type=int, default=SZLAK_INFERENCE_K, dest="inference_k",
                   help="k-NN neighbors at inference (default: 5)")
    p.add_argument("--noise-scale", type=float, required=True, dest="noise_scale",
                   help="ES noise scale relative to vector norm")
    p.add_argument("--limit", type=int, required=True,
                   help="Evaluation samples per iteration")
    p.add_argument("--output", type=str, default="best_transport_rl.pt",
                   help="Output path for best steering object (default: best_transport_rl.pt)")
    p.add_argument("--device", type=str, default=None, help="Compute device (default: auto)")


def _setup_continual_learning_parser(steering_subparsers):
    """Set up the continual learning optimization subparser."""
    p = steering_subparsers.add_parser(
        "continual",
        help="Autonomous continual learning: optimize steering across "
             "multiple tasks with EWC forgetting prevention",
    )
    p.add_argument("model", type=str, help="Model name or path")
    p.add_argument("--tasks", type=str, default=None,
                   help="Comma-separated task list (default: all benchmarks)")
    p.add_argument("--method", type=str, default=None,
                   help="Steering method (default: zwiad-recommended per task)")
    p.add_argument("--max-cycles", type=int, required=True, dest="max_cycles",
                   help="Maximum optimization cycles")
    p.add_argument("--enriched-pairs-dir", type=str, default="./pairs/",
                   dest="enriched_pairs_dir", help="Dir for enriched pairs (default: ./pairs/)")
    p.add_argument("--checkpoint-dir", type=str, default="./checkpoints/",
                   dest="checkpoint_dir", help="Checkpoint directory (default: ./checkpoints/)")
    p.add_argument("--ewc-lambda", type=float, default=CONTINUAL_EWC_LAMBDA, dest="ewc_lambda",
                   help="EWC penalty strength (default: 1000)")
    p.add_argument("--replay-size", type=int, required=True, dest="replay_size",
                   help="Number of past experiences to replay")
    p.add_argument("--replay-interval", type=int, default=CONTINUAL_REPLAY_INTERVAL, dest="replay_interval",
                   help="Cycles between replay checks (default: 5)")
    p.add_argument("--forgetting-threshold", type=float, default=CONTINUAL_FORGETTING_THRESHOLD,
                   dest="forgetting_threshold",
                   help="Score ratio below which forgetting is detected (default: 0.9)")
    p.add_argument("--convergence-window", type=int, default=CONTINUAL_CONVERGENCE_WINDOW,
                   dest="convergence_window",
                   help="Cycles without improvement before stopping (default: 10)")
    p.add_argument("--limit", type=int, required=True,
                   help="Evaluation samples per task")
    p.add_argument("--device", type=str, default=None, help="Compute device (default: auto)")
    p.add_argument("--gcs-bucket", type=str, default=None, dest="gcs_bucket",
                   help="GCS bucket for checkpoint upload (optional)")


def setup_steering_optimizer_parser(parser):
    """Set up the optimize-steering command parser."""
    steering_subparsers = parser.add_subparsers(
        dest="steering_action", help="Steering optimization actions"
    )
    setup_comprehensive_parser(steering_subparsers)
    setup_method_parsers(steering_subparsers)
    setup_personalization_parsers(steering_subparsers)
    setup_welfare_universal_parsers(steering_subparsers)
    _setup_transport_rl_parser(steering_subparsers)
    _setup_continual_learning_parser(steering_subparsers)
