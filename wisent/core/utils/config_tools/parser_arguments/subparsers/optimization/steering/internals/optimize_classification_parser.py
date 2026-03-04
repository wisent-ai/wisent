"""Parser setup for the 'optimize-classification' command."""



def setup_classification_optimizer_parser(parser):
    """Set up the classification-optimizer subcommand parser."""
    parser.add_argument("model", type=str, help="Model name or path to optimize")
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Specific tasks to optimize (e.g., 'truthfulqa_mc1 arc_easy'). If not provided, runs all supported tasks.",
    )
    parser.add_argument("--limit", type=int, required=True, help="Maximum samples per task")
    parser.add_argument(
        "--optimization-metric",
        type=str,
        required=True,
        choices=["f1", "accuracy", "precision", "recall"],
        help="Metric to optimize",
    )
    parser.add_argument(
        "--max-time-per-task", type=float, default=None, help="Maximum time per task in minutes"
    )
    parser.add_argument(
        "--layer-range", type=str, default=None, help="Layer range to test (e.g., '10-20', if None uses all layers)"
    )
    parser.add_argument(
        "--aggregation-methods",
        type=str,
        nargs="+",
        default=None,
        help="Token aggregation methods to test",
    )
    parser.add_argument(
        "--threshold-range",
        type=float,
        nargs="+",
        required=True,
        help="Detection thresholds to test",
    )
    parser.add_argument("--device", type=str, default=None, help="Device to run on")
    parser.add_argument("--results-file", type=str, default=None, help="Custom file path for saving results")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to model config")
    parser.add_argument("--save-logs-json", type=str, default=None, help="Save detailed optimization logs to JSON file")
    parser.add_argument(
        "--save-classifiers",
        action="store_true",
        default=True,
        help="Save best classifiers for each task (default: True)",
    )
    parser.add_argument(
        "--no-save-classifiers",
        dest="save_classifiers",
        action="store_false",
        help="Don't save classifiers (overrides --save-classifiers)",
    )
    parser.add_argument(
        "--classifiers-dir",
        type=str,
        default=None,
        help="Directory to save classifiers (default: ./optimized_classifiers/model_name/)",
    )

    # Timing calibration options
    parser.add_argument(
        "--skip-timing-estimation", action="store_true", help="Skip timing estimation and proceed without time warnings"
    )
    parser.add_argument("--calibration-file", type=str, default=None, help="File to save/load calibration data")
    parser.add_argument(
        "--calibrate-only",
        action="store_true",
        help="Only run calibration and exit (saves to --calibration-file if provided)",
    )

    # Comparison options
    parser.add_argument(
        "--show-comparisons",
        type=int,
        default=0,
        metavar="N",
        help="Show N sample comparisons where optimized config differs from default config. Default: 0 (disabled)",
    )
    parser.add_argument(
        "--save-comparisons",
        type=str,
        default=None,
        metavar="PATH",
        help="Save all sample comparisons to JSON file",
    )

    # Training data options
    parser.add_argument(
        "--use-contrastive-pairs",
        action="store_true",
        help="Augment training with contrastive pairs (ground truth positive/negative responses) in addition to model generations",
    )
    parser.add_argument(
        "--train-on-contrastive-only",
        action="store_true",
        default=True,
        help="Train ONLY on contrastive pairs (no generation for training), but still evaluate on actual model generations. Much faster. (default: True)",
    )
    parser.add_argument(
        "--train-on-generations",
        action="store_true",
        help="Train on actual model generations instead of contrastive pairs (slower, disables --train-on-contrastive-only)",
    )

    # Personalization evaluator parameters
    parser.add_argument(
        "--fast-diversity-seed", type=int, required=True,
        help="Seed for fast diversity computation"
    )
    parser.add_argument(
        "--diversity-max-sample-size", type=int, required=True,
        help="Maximum sample size for diversity computation"
    )
    parser.add_argument(
        "--min-sentence-length", type=int, required=True,
        help="Minimum sentence length for coherence evaluation"
    )
    parser.add_argument(
        "--nonsense-min-tokens", type=int, required=True,
        help="Minimum token count for nonsense word detection"
    )
    parser.add_argument(
        "--quality-min-response-length", type=int, required=True,
        help="Minimum response length for quality scoring"
    )
    parser.add_argument(
        "--quality-repetition-ratio-threshold", type=float, required=True,
        help="Threshold for repetitive token ratio penalty"
    )
    parser.add_argument(
        "--quality-bigram-repeat-threshold", type=int, required=True,
        help="Threshold for repeated bigram count penalty"
    )
    parser.add_argument(
        "--quality-bigram-repeat-penalty", type=float, required=True,
        help="Penalty multiplier for repeated bigrams"
    )
    parser.add_argument(
        "--quality-special-char-ratio-threshold", type=float, required=True,
        help="Threshold for special character ratio penalty"
    )
    parser.add_argument(
        "--quality-special-char-penalty", type=float, required=True,
        help="Penalty multiplier for excessive special characters"
    )
    parser.add_argument(
        "--quality-char-repeat-count", type=int, required=True,
        help="Minimum consecutive character repeats to trigger penalty"
    )
    parser.add_argument(
        "--quality-char-repeat-penalty", type=float, required=True,
        help="Penalty multiplier for character repetition"
    )
