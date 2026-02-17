"""Argparse setup for the migrate-activations CLI command."""


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
        "--strategy", type=str, default="completion_last",
        help="Extraction strategy (default: completion_last)",
    )
    single_parser.add_argument(
        "--database-url", type=str, default=None,
        help="Database URL (defaults to DATABASE_URL env var)",
    )
    single_parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be uploaded without uploading",
    )

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
        "--prompt-format", type=str, default="chat",
        help="Prompt format (default: chat)",
    )
    raw_parser.add_argument(
        "--database-url", type=str, default=None,
        help="Database URL (defaults to DATABASE_URL env var)",
    )
    raw_parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be uploaded without uploading",
    )

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
        "--strategy", type=str, default="completion_last",
        help="Extraction strategy (default: completion_last)",
    )
    verify_parser.add_argument(
        "--layer", type=int, required=True,
        help="Layer number to verify",
    )
    verify_parser.add_argument(
        "--database-url", type=str, default=None,
        help="Database URL (defaults to DATABASE_URL env var)",
    )
