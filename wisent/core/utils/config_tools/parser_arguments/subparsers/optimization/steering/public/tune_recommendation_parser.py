"""Parser setup for the 'tune-recommendation' command."""



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
                     required=True,
                     help="Output JSON path")
    cgt.add_argument("--zwiad-dir", type=str,
                     required=True,
                     help="Directory containing zwiad JSON files")
    cgt.add_argument("--limit", type=int, default=None,
                     help="Max samples per benchmark (default: 100)")
    cgt.add_argument("--device", type=str, default=None,
                     help="Device (cuda, mps, cpu)")
    cgt.add_argument("--methods", type=str, default=None,
                     help="Comma-separated methods to run "
                          "(default: all 7 pipeline methods)")
    cgt.add_argument("--n-trials", type=int, default=None,
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
    cgt.add_argument("--per-type", type=int, default=None,
                     help="Benchmarks per geometry type (default: 2)")

    cgt.add_argument("--lr-lower-bound", type=float, required=True,
                     dest="lr_lower_bound",
                     help="Lower bound for learning rate Optuna search")
    cgt.add_argument("--lr-upper-bound", type=float, required=True,
                     dest="lr_upper_bound",
                     help="Upper bound for learning rate Optuna search")
    cgt.add_argument("--alpha-lower-bound", type=float, required=True,
                     dest="alpha_lower_bound",
                     help="Lower bound for alpha Optuna search")
    cgt.add_argument("--alpha-upper-bound", type=float, required=True,
                     dest="alpha_upper_bound",
                     help="Upper bound for alpha Optuna search")
    cgt.add_argument("--optuna-szlak-reg-min", type=float, required=True,
                     dest="optuna_szlak_reg_min",
                     help="SZLAK: minimum sinkhorn regularization for Optuna search")
    cgt.add_argument("--optuna-nurt-steps-min", type=int, required=True,
                     dest="optuna_nurt_steps_min",
                     help="NURT: minimum integration steps for Optuna search")
    cgt.add_argument("--optuna-nurt-steps-max", type=int, required=True,
                     dest="optuna_nurt_steps_max",
                     help="NURT: maximum integration steps for Optuna search")
    cgt.add_argument("--optuna-wicher-concept-dims", type=int, nargs="+", required=True,
                     dest="optuna_wicher_concept_dims",
                     help="WICHER: concept dimension choices for Optuna search")
    cgt.add_argument("--optuna-wicher-steps-min", type=int, required=True,
                     dest="optuna_wicher_steps_min",
                     help="WICHER: minimum steps for Optuna search")
    cgt.add_argument("--optuna-wicher-steps-max", type=int, required=True,
                     dest="optuna_wicher_steps_max",
                     help="WICHER: maximum steps for Optuna search")
    cgt.add_argument("--optuna-przelom-target-modes", type=str, nargs="+", required=True,
                     dest="optuna_przelom_target_modes",
                     help="PRZELOM: target mode choices for Optuna search")
    cgt.add_argument("--optuna-grom-gate-dim-min", type=int, required=True,
                     dest="optuna_grom_gate_dim_min",
                     help="GROM: minimum gate hidden dim for Optuna search")
    cgt.add_argument("--optuna-grom-gate-dim-max", type=int, required=True,
                     dest="optuna_grom_gate_dim_max",
                     help="GROM: maximum gate hidden dim for Optuna search")
    cgt.add_argument("--optuna-grom-intensity-dim-min", type=int, required=True,
                     dest="optuna_grom_intensity_dim_min",
                     help="GROM: minimum intensity hidden dim for Optuna search")
    cgt.add_argument("--optuna-grom-intensity-dim-max", type=int, required=True,
                     dest="optuna_grom_intensity_dim_max",
                     help="GROM: maximum intensity hidden dim for Optuna search")
    cgt.add_argument("--optuna-grom-sparse-weight-min", type=float, required=True,
                     dest="optuna_grom_sparse_weight_min",
                     help="GROM: minimum sparse weight for Optuna search")
    cgt.add_argument("--optuna-grom-sparse-weight-max", type=float, required=True,
                     dest="optuna_grom_sparse_weight_max",
                     help="GROM: maximum sparse weight for Optuna search")

    # -- optimize-config (CPU) --
    opt = subs.add_parser(
        "optimize-config",
        help="Tune recommendation thresholds and weights using "
             "Optuna on collected ground truth")
    opt.add_argument("--ground-truth", type=str, required=True,
                     help="Path to ground truth JSON from "
                          "collect-ground-truth")
    opt.add_argument("--n-trials", type=int, default=None,
                     help="Number of Optuna trials (default: 500)")
    opt.add_argument("--output", type=str, default=None,
                     help="Output config JSON path "
                          "(default: ~/.wisent/"
                          "learned_recommendation_config.json)")
    opt.add_argument("--objective", type=str, required=True,
                     choices=["top1", "topk", "regret"],
                     help="Objective function")
    opt.add_argument("--top-k", type=int, default=None,
                     help="K for topk objective (default: 2)")
    opt.add_argument("--seed", type=int, default=None,
                     help="Random seed (default: 42)")
