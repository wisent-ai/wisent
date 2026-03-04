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
    p.add_argument("--method", type=str, required=True,
                   help="Steering method. Transport methods "
                        "(przelom, szlak) use cost-matrix REINFORCE; all others use vector ES")
    p.add_argument("--max-iterations", type=int, required=True, dest="max_iterations",
                   help="RL iteration count")
    p.add_argument("--learning-rate", type=float, required=True, dest="learning_rate",
                   help="Step size for cost shaping or ES update")
    p.add_argument("--epsilon", type=float, required=True,
                   help="Entropic regularization temperature for transport methods")
    p.add_argument("--regularization", type=float, default=None,
                   help="Tikhonov regularization for pseudoinverse (required)")
    p.add_argument("--inference-k", type=int, default=None, dest="inference_k",
                   help="k-NN neighbors at inference (required)")
    p.add_argument("--noise-scale", type=float, required=True, dest="noise_scale",
                   help="ES noise scale relative to vector norm")
    p.add_argument("--limit", type=int, required=True,
                   help="Evaluation samples per iteration")
    p.add_argument("--output", type=str, required=True,
                   help="Output path for best steering object")
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
    p.add_argument("--enriched-pairs-dir", type=str, required=True,
                   dest="enriched_pairs_dir", help="Dir for enriched pairs")
    p.add_argument("--checkpoint-dir", type=str, required=True,
                   dest="checkpoint_dir", help="Checkpoint directory")
    p.add_argument("--ewc-lambda", type=float, default=None, dest="ewc_lambda",
                   help="EWC penalty strength")
    p.add_argument("--replay-size", type=int, required=True, dest="replay_size",
                   help="Number of past experiences to replay")
    p.add_argument("--replay-interval", type=int, default=None, dest="replay_interval",
                   help="Cycles between replay checks (default: 5)")
    p.add_argument("--forgetting-threshold", type=float, default=None,
                   dest="forgetting_threshold",
                   help="Score ratio below which forgetting is detected")
    p.add_argument("--convergence-window", type=int, default=None,
                   dest="convergence_window",
                   help="Cycles without improvement before stopping")
    p.add_argument("--epsilon", type=float, required=True,
                   help="Entropic regularization temperature for transport methods")
    p.add_argument("--limit", type=int, required=True,
                   help="Evaluation samples per task")
    p.add_argument("--device", type=str, default=None, help="Compute device (default: auto)")
    p.add_argument("--gcs-bucket", type=str, default=None, dest="gcs_bucket",
                   help="GCS bucket for checkpoint upload (optional)")


def setup_steering_optimizer_parser(parser):
    """Set up the optimize-steering command parser."""
    # Parent-level arguments for the default Optuna path
    parser.add_argument("--lr-lower-bound", type=float, default=None,
                        dest="lr_lower_bound",
                        help="Lower bound for learning rate Optuna search")
    parser.add_argument("--lr-upper-bound", type=float, default=None,
                        dest="lr_upper_bound",
                        help="Upper bound for learning rate Optuna search")
    parser.add_argument("--alpha-lower-bound", type=float, default=None,
                        dest="alpha_lower_bound",
                        help="Lower bound for alpha Optuna search")
    parser.add_argument("--alpha-upper-bound", type=float, default=None,
                        dest="alpha_upper_bound",
                        help="Upper bound for alpha Optuna search")
    parser.add_argument("--optuna-szlak-reg-min", type=float, required=True,
                        dest="optuna_szlak_reg_min",
                        help="SZLAK: minimum sinkhorn regularization for Optuna search")
    parser.add_argument("--optuna-nurt-steps-min", type=int, required=True,
                        dest="optuna_nurt_steps_min",
                        help="NURT: minimum integration steps for Optuna search")
    parser.add_argument("--optuna-nurt-steps-max", type=int, required=True,
                        dest="optuna_nurt_steps_max",
                        help="NURT: maximum integration steps for Optuna search")
    parser.add_argument("--optuna-wicher-concept-dims", type=int, nargs="+", required=True,
                        dest="optuna_wicher_concept_dims",
                        help="WICHER: concept dimension choices for Optuna search")
    parser.add_argument("--optuna-wicher-steps-min", type=int, required=True,
                        dest="optuna_wicher_steps_min",
                        help="WICHER: minimum steps for Optuna search")
    parser.add_argument("--optuna-wicher-steps-max", type=int, required=True,
                        dest="optuna_wicher_steps_max",
                        help="WICHER: maximum steps for Optuna search")
    parser.add_argument("--optuna-przelom-target-modes", type=str, nargs="+", required=True,
                        dest="optuna_przelom_target_modes",
                        help="PRZELOM: target mode choices for Optuna search")
    parser.add_argument("--optuna-grom-gate-dim-min", type=int, required=True,
                        dest="optuna_grom_gate_dim_min",
                        help="GROM: minimum gate hidden dim for Optuna search")
    parser.add_argument("--optuna-grom-gate-dim-max", type=int, required=True,
                        dest="optuna_grom_gate_dim_max",
                        help="GROM: maximum gate hidden dim for Optuna search")
    parser.add_argument("--optuna-grom-intensity-dim-min", type=int, required=True,
                        dest="optuna_grom_intensity_dim_min",
                        help="GROM: minimum intensity hidden dim for Optuna search")
    parser.add_argument("--optuna-grom-intensity-dim-max", type=int, required=True,
                        dest="optuna_grom_intensity_dim_max",
                        help="GROM: maximum intensity hidden dim for Optuna search")
    parser.add_argument("--optuna-grom-sparse-weight-min", type=float, required=True,
                        dest="optuna_grom_sparse_weight_min",
                        help="GROM: minimum sparse weight for Optuna search")
    parser.add_argument("--optuna-grom-sparse-weight-max", type=float, required=True,
                        dest="optuna_grom_sparse_weight_max",
                        help="GROM: maximum sparse weight for Optuna search")

    steering_subparsers = parser.add_subparsers(
        dest="steering_action", help="Steering optimization actions"
    )
    setup_comprehensive_parser(steering_subparsers)
    setup_method_parsers(steering_subparsers)
    setup_personalization_parsers(steering_subparsers)
    setup_welfare_universal_parsers(steering_subparsers)
    _setup_transport_rl_parser(steering_subparsers)
    _setup_continual_learning_parser(steering_subparsers)
