"""Argparse setup for the migrate-activations CLI command."""


def _add_hf_retry_args(p):
    """Add HF upload retry arguments to a subparser."""
    p.add_argument("--hf-max-retries", type=int, required=True, help="Max upload retries")
    p.add_argument("--hf-base-wait", type=int, required=True, help="Base wait seconds between retries")
    p.add_argument("--hf-backoff-max-exp", type=int, required=True, help="Max exponent for backoff")
    p.add_argument("--hf-jitter-min", type=float, required=True, help="Minimum jitter multiplier")
    p.add_argument("--hf-jitter-max", type=float, required=True, help="Maximum jitter multiplier")
    p.add_argument("--hf-retryable-pattern", type=str, action="append", required=True, help="Retryable error pattern (repeatable)")


def setup_migrate_activations_parser(parser):
    """Configure the migrate-activations subparser."""
    subparsers = parser.add_subparsers(dest="action", help="Migration actions")

    # create-repo
    subparsers.add_parser(
        "create-repo",
        help="Create the wisent-ai/activations HF dataset repo",
    )

    # all - migrate everything
    all_parser = subparsers.add_parser(
        "all", help="Migrate all activations from Supabase to HF"
    )
    all_parser.add_argument(
        "--database-url", type=str, default=None,
        help="Database URL (defaults to DATABASE_URL env var)",
    )
    all_parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be uploaded without uploading",
    )
    all_parser.add_argument(
        "--combo-start", type=int, default=0,
        help="First combo index (inclusive, default: 0)",
    )
    all_parser.add_argument(
        "--combo-end", type=int, default=None,
        help="Last combo index (exclusive, default: all)",
    )
    all_parser.add_argument(
        "--skip-pair-texts", action="store_true",
        help="Skip pair texts migration",
    )
    all_parser.add_argument(
        "--reverse", action="store_true",
        help="Process combos in reverse order within the slice",
    )
    _add_hf_retry_args(all_parser)

    # consolidate - build index.json from markers
    consolidate_parser = subparsers.add_parser(
        "consolidate",
        help="Build unified index.json from marker files",
    )
    consolidate_parser.add_argument(
        "--dry-run", action="store_true",
        help="Print consolidated index without uploading",
    )
    _add_hf_retry_args(consolidate_parser)

    # single - migrate one (model, benchmark, strategy)
    single_parser = subparsers.add_parser(
        "single", help="Migrate a single model/benchmark/strategy combo"
    )
    single_parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model ID",
    )
    single_parser.add_argument(
        "--benchmark", type=str, required=True,
        help="Benchmark/task name",
    )
    single_parser.add_argument(
        "--strategy", type=str, required=True,
        help="Extraction strategy",
    )
    single_parser.add_argument(
        "--database-url", type=str, default=None,
        help="Database URL (defaults to DATABASE_URL env var)",
    )
    single_parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be uploaded without uploading",
    )
    _add_hf_retry_args(single_parser)

    # pair-texts - migrate pair texts for a benchmark
    pt_parser = subparsers.add_parser(
        "pair-texts", help="Migrate pair texts for a benchmark"
    )
    pt_parser.add_argument(
        "--benchmark", type=str, required=True,
        help="Benchmark/task name",
    )
    pt_parser.add_argument(
        "--database-url", type=str, default=None,
        help="Database URL (defaults to DATABASE_URL env var)",
    )
    pt_parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be uploaded without uploading",
    )
    _add_hf_retry_args(pt_parser)

    # raw - migrate raw activations
    raw_parser = subparsers.add_parser(
        "raw", help="Migrate raw activations (chunked)"
    )
    raw_parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model ID",
    )
    raw_parser.add_argument(
        "--benchmark", type=str, required=True,
        help="Benchmark/task name",
    )
    raw_parser.add_argument(
        "--prompt-format", type=str, required=True,
        help="Prompt format",
    )
    raw_parser.add_argument(
        "--chunk-size", type=int, required=True,
        help="Number of pairs per shard for raw activation migration",
    )
    raw_parser.add_argument(
        "--database-url", type=str, default=None,
        help="Database URL (defaults to DATABASE_URL env var)",
    )
    raw_parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be uploaded without uploading",
    )
    _add_hf_retry_args(raw_parser)

    # verify-completeness - check all DB combos exist in HF
    vc_parser = subparsers.add_parser(
        "verify-completeness",
        help="Check all Supabase activation combos exist in HF index",
    )
    vc_parser.add_argument(
        "--database-url", type=str, default=None,
        help="Database URL (defaults to DATABASE_URL env var)",
    )
    _add_hf_retry_args(vc_parser)

    # verify - verify migration for a single layer
    verify_parser = subparsers.add_parser(
        "verify", help="Verify HF data matches Supabase"
    )
    verify_parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model ID",
    )
    verify_parser.add_argument(
        "--benchmark", type=str, required=True,
        help="Benchmark/task name",
    )
    verify_parser.add_argument(
        "--strategy", type=str, required=True,
        help="Extraction strategy",
    )
    verify_parser.add_argument(
        "--layer", type=int, required=True,
        help="Layer number to verify",
    )
    verify_parser.add_argument(
        "--database-url", type=str, default=None,
        help="Database URL (defaults to DATABASE_URL env var)",
    )
