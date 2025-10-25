"""Parser setup for the 'optimize-classification' command."""


def setup_classification_optimizer_parser(parser):
    """Set up the classification-optimizer subcommand parser."""
    parser.add_argument("model", type=str, help="Model name or path to optimize")
    parser.add_argument("--limit", type=int, default=1000, help="Maximum samples per task (default: 1000)")
    parser.add_argument(
        "--optimization-metric",
        type=str,
        default="f1",
        choices=["f1", "accuracy", "precision", "recall"],
        help="Metric to optimize (default: f1)",
    )
    parser.add_argument(
        "--max-time-per-task", type=float, default=15.0, help="Maximum time per task in minutes (default: 15.0)"
    )
    parser.add_argument(
        "--layer-range", type=str, default=None, help="Layer range to test (e.g., '10-20', if None uses all layers)"
    )
    parser.add_argument(
        "--aggregation-methods",
        type=str,
        nargs="+",
        default=["average", "final", "first", "max", "min"],
        help="Token aggregation methods to test",
    )
    parser.add_argument(
        "--threshold-range",
        type=float,
        nargs="+",
        default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
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
