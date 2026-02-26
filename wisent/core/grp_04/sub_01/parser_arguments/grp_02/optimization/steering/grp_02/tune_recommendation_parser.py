"""Parser setup for the 'tune-recommendation' command."""

from wisent.core.constants import (
    BENCHMARKS_PER_TYPE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_N_TRIALS,
    OPTUNA_TRIALS_TUNE,
    TOPK_OBJECTIVE_K,
)


def setup_tune_recommendation_parser(parser):
    """Add two subcommands: collect-ground-truth and optimize-config."""
    subs = parser.add_subparsers(dest="subcommand")

    # -- collect-ground-truth (GPU) --
    cgt = subs.add_parser(
        "collect-ground-truth",
        help="Run all steering methods on benchmarks to collect "
             "ground-truth accuracy data")
    cgt.add_argument("--model", type=str, required=True,
                     help="HuggingFace model name or path")
    cgt.add_argument("--benchmarks", type=str, default=None,
                     help="Comma-separated benchmark names "
                          "(default: all with zwiad results)")
    cgt.add_argument("--output", type=str,
                     default="ground_truth.json",
                     help="Output JSON path (default: ground_truth.json)")
    cgt.add_argument("--zwiad-dir", type=str,
                     default="zwiad_results",
                     help="Directory containing zwiad JSON files")
    cgt.add_argument("--limit", type=int, default=DEFAULT_N_TRIALS,
                     help="Max samples per benchmark (default: 100)")
    cgt.add_argument("--device", type=str, default=None,
                     help="Device (cuda, mps, cpu)")
    cgt.add_argument("--methods", type=str, default=None,
                     help="Comma-separated methods to run "
                          "(default: all 7 pipeline methods)")
    cgt.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS,
                     help="Optuna trials per method (default: 100)")
    cgt.add_argument("--benchmark-start", type=int, default=None,
                     help="Start index for benchmark sharding")
    cgt.add_argument("--benchmark-end", type=int, default=None,
                     help="End index for benchmark sharding")
    cgt.add_argument("--use-geometry-selection", action="store_true",
                     help="Select representative benchmarks per "
                          "geometry type instead of running all")
    cgt.add_argument("--fine-geometry", action="store_true",
                     help="Use 8-type fine geometry (default: 5-type)")
    cgt.add_argument("--per-type", type=int, default=BENCHMARKS_PER_TYPE,
                     help="Benchmarks per geometry type (default: 2)")

    # -- optimize-config (CPU) --
    opt = subs.add_parser(
        "optimize-config",
        help="Tune recommendation thresholds and weights using "
             "Optuna on collected ground truth")
    opt.add_argument("--ground-truth", type=str, required=True,
                     help="Path to ground truth JSON from "
                          "collect-ground-truth")
    opt.add_argument("--n-trials", type=int, default=OPTUNA_TRIALS_TUNE,
                     help="Number of Optuna trials (default: 500)")
    opt.add_argument("--output", type=str, default=None,
                     help="Output config JSON path "
                          "(default: ~/.wisent/"
                          "learned_recommendation_config.json)")
    opt.add_argument("--objective", type=str, default="top1",
                     choices=["top1", "topk", "regret"],
                     help="Objective function (default: top1)")
    opt.add_argument("--top-k", type=int, default=TOPK_OBJECTIVE_K,
                     help="K for topk objective (default: 2)")
    opt.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED,
                     help="Random seed (default: 42)")
