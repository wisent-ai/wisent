"""Parser setup for the 'generate-pairs' command."""


def setup_generate_pairs_parser(parser):
    """Set up the generate-pairs subcommand parser."""
    parser.add_argument(
        "--trait", type=str, required=True, help="Natural language description of the desired trait or behavior"
    )
    parser.add_argument(
        "--num-pairs", type=int, default=30, help="Number of contrastive pairs to generate (default: 30)"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output file path for the generated pairs (JSON format)"
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or path to use for generation"
    )
    parser.add_argument("--device", type=str, default=None, help="Device to run on")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for deduplication (0-1, higher = more strict)",
    )
    parser.add_argument("--timing", action="store_true", help="Show detailed timing for each generation step")
    parser.add_argument(
        "--max-workers", type=int, default=4, help="Number of parallel workers for generation (default: 4)"
    )
