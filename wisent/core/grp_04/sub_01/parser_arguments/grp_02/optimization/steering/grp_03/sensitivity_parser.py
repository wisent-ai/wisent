"""Parser setup for the 'sensitivity' command.

Three subcommands:
  run       -- One-at-a-time sensitivity sweep across constant registry
  report    -- View ranked sensitivity results from saved JSON
  optimize  -- Optuna joint optimization of top-N most sensitive constants
  calibrate -- Threshold calibration from empirical metric distributions
"""

from wisent.core.constants import (
    DEFAULT_LIMIT,
    DEFAULT_N_TRIALS,
    DEFAULT_RANDOM_SEED,
    PARSER_DEFAULT_NUM_STRENGTH_STEPS,
    SENSITIVITY_DEFAULT_THRESHOLD,
    SENSITIVITY_TOP_CONSTANTS,
)


def setup_sensitivity_parser(parser):
    """Set up the sensitivity command parser with subcommands."""
    subs = parser.add_subparsers(dest="sensitivity_action")

    _setup_run_parser(subs)
    _setup_report_parser(subs)
    _setup_optimize_parser(subs)
    _setup_calibrate_parser(subs)


def _setup_run_parser(subs):
    """Set up the 'run' subcommand for sensitivity sweep."""
    run_p = subs.add_parser(
        "run",
        help="Run one-at-a-time sensitivity sweep across "
             "registered constants, measuring score delta",
    )
    run_p.add_argument(
        "model", type=str,
        help="HuggingFace model name or path",
    )
    run_p.add_argument(
        "--task", type=str, required=True,
        help="Benchmark task name for evaluation",
    )
    run_p.add_argument(
        "--method", type=str, required=True,
        help="Steering method name (e.g. grom, tecza, tetno)",
    )
    run_p.add_argument(
        "--group", type=str, default=None,
        choices=["D", "E"],
        help="Filter by constant group: D (ML hyperparams) "
             "or E (thresholds). Default: both.",
    )
    run_p.add_argument(
        "--method-filter", type=str, default=None,
        dest="method_filter",
        help="Filter constants by method name (e.g. grom, evaluator)",
    )
    run_p.add_argument(
        "--steps", type=int, default=PARSER_DEFAULT_NUM_STRENGTH_STEPS,
        help="Number of values to test per constant (default: 5)",
    )
    run_p.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help="Max samples for evaluation (default: 100)",
    )
    run_p.add_argument(
        "--output", type=str, default="sensitivity_result.json",
        help="Output JSON path (default: sensitivity_result.json)",
    )
    run_p.add_argument(
        "--device", type=str, default=None,
        help="Compute device (cuda, mps, cpu, auto)",
    )
    run_p.add_argument(
        "--seed", type=int, default=DEFAULT_RANDOM_SEED,
        help="Random seed (default: 42)",
    )


def _setup_report_parser(subs):
    """Set up the 'report' subcommand for viewing results."""
    report_p = subs.add_parser(
        "report",
        help="Display ranked sensitivity results from "
             "a saved JSON file",
    )
    report_p.add_argument(
        "--input", type=str, required=True,
        help="Path to sensitivity result JSON",
    )
    report_p.add_argument(
        "--threshold", type=float, default=SENSITIVITY_DEFAULT_THRESHOLD,
        help="Minimum sensitivity to display (default: 0.01)",
    )
    report_p.add_argument(
        "--top-n", type=int, default=None,
        dest="top_n",
        help="Show only top-N most sensitive constants",
    )


def _setup_optimize_parser(subs):
    """Set up the 'optimize' subcommand for Optuna optimization."""
    opt_p = subs.add_parser(
        "optimize",
        help="Jointly optimize the top-N most sensitive "
             "constants via Optuna TPE sampler",
    )
    opt_p.add_argument(
        "model", type=str,
        help="HuggingFace model name or path",
    )
    opt_p.add_argument(
        "--task", type=str, required=True,
        help="Benchmark task name for evaluation",
    )
    opt_p.add_argument(
        "--method", type=str, required=True,
        help="Steering method name (e.g. grom, tecza)",
    )
    opt_p.add_argument(
        "--input", type=str, required=True,
        help="Sensitivity result JSON (from 'sensitivity run')",
    )
    opt_p.add_argument(
        "--top-n", type=int, default=SENSITIVITY_TOP_CONSTANTS,
        dest="top_n",
        help="Number of top constants to optimize (default: 20)",
    )
    opt_p.add_argument(
        "--n-trials", type=int, default=DEFAULT_N_TRIALS,
        dest="n_trials",
        help="Number of Optuna trials (default: 100)",
    )
    opt_p.add_argument(
        "--sensitivity-threshold", type=float, default=0.0,
        dest="sensitivity_threshold",
        help="Minimum sensitivity to include (default: 0.0)",
    )
    opt_p.add_argument(
        "--output", type=str, default=None,
        help="Output profile JSON (default: auto-named in "
             "~/.wisent/constant_profiles/)",
    )
    opt_p.add_argument(
        "--device", type=str, default=None,
        help="Compute device (cuda, mps, cpu, auto)",
    )
    opt_p.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help="Max evaluation samples (default: 100)",
    )


def _setup_calibrate_parser(subs):
    """Set up the 'calibrate' subcommand for threshold calibration."""
    cal_p = subs.add_parser(
        "calibrate",
        help="Calibrate Group E thresholds from empirical "
             "metric distributions on real data",
    )
    cal_p.add_argument(
        "model", type=str,
        help="HuggingFace model name or path",
    )
    cal_p.add_argument(
        "--task", type=str, required=True,
        help="Benchmark task name for data collection",
    )
    cal_p.add_argument(
        "--pairs-file", type=str, default=None,
        dest="pairs_file",
        help="Pre-generated pairs JSON (default: generate from task)",
    )
    cal_p.add_argument(
        "--output", type=str, default=None,
        help="Output calibration JSON (default: auto-named in "
             "~/.wisent/constant_profiles/)",
    )
    cal_p.add_argument(
        "--device", type=str, default=None,
        help="Compute device (cuda, mps, cpu, auto)",
    )
    cal_p.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help="Max pairs for calibration (default: 100)",
    )
