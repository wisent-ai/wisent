"""Parser setup for the 'evidence' command.

Subcommands:
  compare-axis -- Run axis comparison and record to the evidence ledger
  list         -- List evidence records
  report       -- Show search-space reductions for a model
  sync         -- Upload / download ledger to/from GCS
"""
from wisent.core.constants import DEFAULT_LIMIT


def setup_evidence_parser(parser):
    """Set up the evidence command parser with subcommands."""
    subs = parser.add_subparsers(dest="evidence_action")
    _setup_compare_axis(subs)
    _setup_list(subs)
    _setup_report(subs)
    _setup_sync(subs)


def _setup_compare_axis(subs):
    p = subs.add_parser(
        "compare-axis",
        help="Compare values of a single search-space axis and "
             "record the result to the evidence ledger",
    )
    p.add_argument(
        "model", type=str,
        help="HuggingFace model name or path",
    )
    p.add_argument(
        "--task", type=str, required=True,
        help="Benchmark task name for evaluation",
    )
    p.add_argument(
        "--method", type=str, required=True,
        help="Steering method name (e.g. grom, tecza, caa)",
    )
    p.add_argument(
        "--axis", type=str, required=True,
        help="Axis to compare (extraction_strategy, prompt_strategy, "
             "steering_strategy, strength, sensor_layer, "
             "num_directions, gate_hidden_dim)",
    )
    p.add_argument(
        "--values", type=str, default=None,
        help="Comma-separated values to test. Omit for axis defaults.",
    )
    p.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help="Max evaluation samples (default: 100)",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="Compute device (cuda, mps, cpu, auto)",
    )


def _setup_list(subs):
    p = subs.add_parser(
        "list",
        help="List all evidence records in the ledger",
    )
    p.add_argument(
        "--model", type=str, default=None,
        help="Filter by model name",
    )
    p.add_argument(
        "--axis", type=str, default=None,
        help="Filter by axis name",
    )


def _setup_report(subs):
    p = subs.add_parser(
        "report",
        help="Show search-space reductions the evidence ledger "
             "would apply for a given model",
    )
    p.add_argument(
        "model", type=str,
        help="HuggingFace model name to compute reductions for",
    )
    p.add_argument(
        "--task", type=str, default=None,
        help="Filter evidence by task (optional)",
    )
    p.add_argument(
        "--method", type=str, default=None,
        help="Filter evidence by method (optional)",
    )


def _setup_sync(subs):
    p = subs.add_parser(
        "sync",
        help="Synchronise the evidence ledger with GCS",
    )
    p.add_argument(
        "--upload", action="store_true",
        help="Upload local ledger to GCS",
    )
    p.add_argument(
        "--download", action="store_true",
        help="Download ledger from GCS (merge with local)",
    )
