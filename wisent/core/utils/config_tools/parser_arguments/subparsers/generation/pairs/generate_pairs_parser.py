"""Parser setup for the 'generate-pairs' command."""



def setup_generate_pairs_parser(parser):
    """Set up the generate-pairs subcommand parser."""
    parser.add_argument(
        "--trait", type=str, required=True, help="Natural language description of the desired trait or behavior"
    )
    parser.add_argument(
        "--num-pairs", type=int, required=True, help="Number of contrastive pairs to generate"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output file path for the generated pairs (JSON format)"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model name or path to use for generation"
    )
    parser.add_argument("--device", type=str, default=None, help="Device to run on")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        required=True,
        help="Similarity threshold for deduplication (0-1, higher = more strict)",
    )
    parser.add_argument("--timing", action="store_true", help="Show detailed timing for each generation step")
    parser.add_argument(
        "--max-workers", type=int, required=True, help="Number of parallel workers for generation"
    )

    parser.add_argument(
        "--trait-label-max-length", type=int, required=True,
        help="Maximum length for trait label strings",
    )
    parser.add_argument(
        "--trait-name-max-length", type=int, required=True,
        help="Maximum length for trait name strings",
    )

    # Data infrastructure parameters
    parser.add_argument(
        "--generate-pairs-min-tokens", type=int, required=True,
        help="Minimum tokens for pair generation"
    )
    parser.add_argument(
        "--simhash-relaxed-threshold-bits", type=int, required=True,
        help="SimHash relaxed threshold bits for deduplication"
    )
    parser.add_argument(
        "--tokens-per-pair-estimate", type=int, required=True,
        help="Estimated tokens per contrastive pair"
    )
    parser.add_argument(
        "--tokens-base-offset", type=int, required=True,
        help="Base token offset for generation"
    )
    parser.add_argument(
        "--dedup-word-ngram", type=int, required=True,
        help="Word n-gram size for deduplication"
    )
    parser.add_argument(
        "--dedup-char-ngram", type=int, required=True,
        help="Character n-gram size for deduplication"
    )
    parser.add_argument(
        "--simhash-num-bands", type=int, required=True,
        help="Number of bands for SimHash LSH"
    )

    # Nonsense generation options
    parser.add_argument(
        "--nonsense",
        action="store_true",
        help="Generate nonsense contrastive pairs (negative responses are gibberish/nonsense)",
    )
    parser.add_argument(
        "--nonsense-mode",
        type=str,
        choices=["random_chars", "repetitive", "word_salad", "mixed"],
        required=True,
        help="Type of nonsense to generate: 'random_chars' (ahsdhashdahsdha), 'repetitive' (the the the), 'word_salad' (real words, no meaning), 'mixed' (combination)",
    )
    parser.add_argument(
        "--fast-diversity-seed", type=int, required=True,
        help="Random seed for diversity computation"
    )
    parser.add_argument(
        "--diversity-max-sample-size", type=int, required=True,
        help="Maximum sample size for diversity metrics computation"
    )
    parser.add_argument(
        "--retry-multiplier", type=int, required=True,
        help="Multiplier for max generation retry attempts"
    )
    parser.add_argument(
        "--nonsense-default-num-pairs", type=int, required=True,
        help="Default number of nonsense pairs to generate"
    )
