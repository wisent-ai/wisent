"""Parser for the 'repscan' command - geometry analysis with concept decomposition."""


def setup_repscan_parser(parser):
    """
    Set up the repscan command parser.

    Usage:
        wisent repscan --from-database --model meta-llama/Llama-3.2-1B-Instruct --task truthfulqa_custom
        wisent repscan --from-cache cache/optimize/meta-llama_Llama-3.2-1B-Instruct/activations_xxx.pkl
        wisent repscan --from-database --task truthfulqa_custom --layers 8-16 --visualizations
    """
    # Data source - mutually exclusive
    data_source = parser.add_mutually_exclusive_group(required=True)
    data_source.add_argument(
        "--from-database",
        action="store_true",
        help="Load activations from Supabase database"
    )
    data_source.add_argument(
        "--from-cache",
        type=str,
        metavar="PATH",
        help="Load activations from local cache pickle file"
    )

    # Model and task (required only for --from-database)
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model name in database (default: meta-llama/Llama-3.2-1B-Instruct)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task/benchmark name in database (required for --from-database, optional for --from-cache to enable LLM concept naming)"
    )

    # Layer selection
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Layers to analyze: range (8-16) or comma-separated (8,10,12). Default: all available layers in database"
    )

    # Database options
    parser.add_argument(
        "--prompt-format",
        type=str,
        default="chat",
        choices=["chat", "completion"],
        help="Prompt format used for activations (default: chat)"
    )
    parser.add_argument(
        "--extraction-strategy",
        type=str,
        default="last_token",
        choices=["last_token", "first_token"],
        help="Token extraction strategy (default: last_token)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of pairs to load (default: all)"
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="Database URL (default: DATABASE_URL env var)"
    )

    # Protocol steps
    parser.add_argument(
        "--steps",
        type=str,
        default="all",
        help="Protocol steps to run: 'all', 'signal', 'geometry', 'decomposition', 'intervention', 'editability', or comma-separated (e.g., 'signal,editability')"
    )

    # Analysis options
    parser.add_argument(
        "--visualizations",
        action="store_true",
        help="Generate visualization figures"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="LLM model for concept naming (default: Qwen/Qwen3-8B)"
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for JSON results"
    )
    parser.add_argument(
        "--visualizations-dir",
        type=str,
        default="./repscan_visualizations",
        help="Directory to save visualization PNGs (default: ./repscan_visualizations)"
    )
