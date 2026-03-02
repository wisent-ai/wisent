"""Steering and advanced arguments for the tasks parser."""

from wisent.core.utils.config_tools.constants import DEFAULT_STRENGTH, NONSENSE_MAX_WORD_LENGTH, MIN_QUALITY_SCORE_OUTPUT, TRACKING_SAMPLING_INTERVAL, MIN_SNR_EARLY_REJECT, MIN_CV_SCORE_EARLY_REJECT


def setup_steering_task_args(parser):
    """Set up steering, training, optimization, and advanced task arguments."""
    # Mode selection arguments (one is required for task execution)
    parser.add_argument(
        "--steering-mode",
        action="store_true",
        help="STEERING MODE: Train steering vectors to modify model behavior. Uses zwiad to auto-select best method (CAA/GROM/TECZA). Evaluates baseline vs steered accuracy."
    )
    parser.add_argument(
        "--classification-mode",
        action="store_true",
        help="CLASSIFICATION MODE: Train a classifier to detect good/bad responses. Uses activations to predict response quality. Outputs classifier accuracy and F1 score."
    )
    parser.add_argument(
        "--steering-strength", type=float, default=DEFAULT_STRENGTH, help="Strength of steering vector application (default: 1.0)"
    )

    # Steering method selection - uses centralized registry
    from wisent.core.control.steering_methods import SteeringMethodRegistry
    SteeringMethodRegistry.add_all_cli_arguments(parser)

    # Steering output mode selection
    parser.add_argument(
        "--output-mode",
        type=str,
        default="both",
        choices=["likelihoods", "responses", "both"],
        help="Type of comparison to show: 'likelihoods' for log-likelihood comparison only, 'responses' for response generation only, 'both' for both (default: both)",
    )

    # Token steering arguments
    parser.add_argument("--enable-token-steering", action="store_true", help="Enable token-level steering control")
    parser.add_argument(
        "--token-steering-strategy",
        type=str,
        default="last_only",
        choices=[
            "last_only",
            "first_only",
            "all_equal",
            "exponential_decay",
            "exponential_growth",
            "linear_decay",
            "linear_growth",
            "custom",
        ],
        help="Token steering strategy (default: last_only)",
    )
    parser.add_argument(
        "--token-decay-rate",
        type=float,
        required=True,
        help="Decay rate for exponential token steering strategies (0-1)",
    )
    parser.add_argument(
        "--token-min-strength",
        type=float,
        required=True,
        help="Minimum steering strength for token strategies",
    )
    parser.add_argument(
        "--token-max-strength",
        type=float,
        default=DEFAULT_STRENGTH,
        help="Maximum steering strength for token strategies (default: 1.0)",
    )
    parser.add_argument(
        "--token-apply-to-prompt",
        action="store_true",
        help="Apply steering to prompt tokens as well as generated tokens",
    )
    parser.add_argument(
        "--token-prompt-strength-multiplier",
        type=float,
        required=True,
        help="Strength multiplier for prompt tokens",
    )

    # Training/Inference mode arguments
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Training-only mode: train classifiers/vectors and save them, skip inference",
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Inference-only mode: load pre-trained classifiers/vectors and use for monitoring/steering",
    )
    parser.add_argument(
        "--save-classifier",
        type=str,
        default=None,
        help="Path to save trained classifier(s). In multi-layer mode, saves one file per layer with layer suffix",
    )
    parser.add_argument(
        "--load-classifier",
        type=str,
        default=None,
        help="Path to load pre-trained classifier(s). In multi-layer mode, expects files with layer suffix",
    )
    parser.add_argument(
        "--classifier-dir",
        type=str,
        default="./models",
        help="Directory for saving/loading classifiers and vectors (default: ./models)",
    )
    # Normalization options
    parser.add_argument("--normalize-mode", action="store_true", help="Enable normalization mode (legacy flag)")
    parser.add_argument(
        "--normalization-method",
        type=str,
        default="none",
        choices=["none", "l2_unit", "cross_behavior", "layer_wise_mean"],
        help="Vector normalization method to apply",
    )
    parser.add_argument("--target-norm", type=float, default=None, help="Target norm for certain normalization methods")

    # Nonsense detection options
    parser.add_argument(
        "--enable-nonsense-detection",
        action="store_true",
        help="Enable nonsense detection to stop lobotomized responses",
    )
    parser.add_argument(
        "--max-word-length",
        type=int,
        default=NONSENSE_MAX_WORD_LENGTH,
        help="Maximum reasonable word length for nonsense detection (default: 20)",
    )
    parser.add_argument(
        "--repetition-threshold",
        type=float,
        required=True,
        help="Threshold for repetitive content detection (0-1)",
    )
    parser.add_argument(
        "--gibberish-threshold",
        type=float,
        required=True,
        help="Threshold for gibberish word detection (0-1)",
    )
    parser.add_argument(
        "--disable-dictionary-check",
        action="store_true",
        help="Disable dictionary-based word validation (faster but less accurate)",
    )
    parser.add_argument(
        "--nonsense-action",
        type=str,
        default="regenerate",
        choices=["regenerate", "stop", "flag"],
        help="Action when nonsense is detected: regenerate, stop generation, or flag for review",
    )
    parser.add_argument(
        "--enable-quality-check",
        action="store_true",
        help="Enable quality/coherence checking of generated outputs (detects gibberish)",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=MIN_QUALITY_SCORE_OUTPUT,
        help="Minimum quality score (1-100) to consider output acceptable (default: 50.0)",
    )

    # Performance monitoring options
    parser.add_argument(
        "--enable-memory-tracking", action="store_true", help="Enable memory usage tracking and reporting"
    )
    parser.add_argument(
        "--enable-latency-tracking", action="store_true", help="Enable latency/timing tracking and reporting"
    )
    parser.add_argument(
        "--memory-sampling-interval", type=float, default=TRACKING_SAMPLING_INTERVAL, help="Memory sampling interval in seconds (default: 0.1)"
    )
    parser.add_argument("--track-gpu-memory", action="store_true", help="Track GPU memory usage (requires CUDA)")
    parser.add_argument(
        "--detailed-performance-report",
        action="store_true",
        help="Generate detailed performance report with all metrics",
    )
    parser.add_argument("--export-performance-csv", type=str, default=None, help="Export performance data to CSV file")
    parser.add_argument(
        "--show-memory-usage", action="store_true", help="Show current memory usage without full tracking"
    )
    parser.add_argument("--show-timing-summary", action="store_true", help="Show timing summary after evaluation")

    # Test-time activation saving/loading options
    parser.add_argument(
        "--save-test-activations", type=str, default=None, help="Save test activations to file for future use"
    )
    parser.add_argument(
        "--load-test-activations", type=str, default=None, help="Load test activations from file instead of computing"
    )

    # Priority-aware benchmark selection options
    parser.add_argument(
        "--priority",
        type=str,
        default="all",
        choices=["all", "high", "medium", "low"],
        help="Priority level for benchmark selection (default: all)",
    )
    parser.add_argument(
        "--fast-only", action="store_true", help="Only use fast benchmarks (high priority, < 13.5s loading time)"
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=None,
        help="Time budget in minutes for benchmark selection (auto-selects fast benchmarks)",
    )
    parser.add_argument(
        "--max-benchmarks",
        type=int,
        default=None,
        help="Maximum number of benchmarks to select (combines with priority filtering)",
    )
    parser.add_argument(
        "--smart-selection", action="store_true", help="Use smart benchmark selection based on relevance and priority"
    )
    parser.add_argument(
        "--prefer-fast",
        action="store_true",
        help="Prefer fast benchmarks in selection when multiple options are available",
    )
    parser.add_argument(
        "--save-steering-vector", type=str, default=None, help="Path to save the computed steering vector"
    )
    parser.add_argument(
        "--load-steering-vector", type=str, default=None, help="Path to load a pre-computed steering vector"
    )
    parser.add_argument(
        "--accept-low-quality-vector",
        action="store_true",
        default=False,
        help="Accept steering vectors that fail quality checks (convergence, SNR, etc.)"
    )
    # Early rejection during optimization
    parser.add_argument(
        "--disable-early-rejection",
        action="store_true",
        default=False,
        help="Disable early rejection of low-quality vectors during optimization (slower but explores more)"
    )
    parser.add_argument(
        "--early-rejection-snr-threshold",
        type=float,
        default=MIN_SNR_EARLY_REJECT,
        help="Minimum SNR for early rejection during optimization (default: 5.0)"
    )
    parser.add_argument(
        "--early-rejection-cv-threshold",
        type=float,
        default=MIN_CV_SCORE_EARLY_REJECT,
        help="Minimum cross-validation score for early rejection during optimization (default: 0.1)"
    )

    # Additional output options
    parser.add_argument("--csv-output", type=str, default=None, help="Path to save results in CSV format")
    parser.add_argument("--evaluation-report", type=str, default=None, help="Path to save evaluation report")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue processing other tasks if one fails")

    # Benchmark caching arguments
    parser.add_argument(
        "--cache-benchmark",
        action="store_true",
        default=True,
        help="Cache the benchmark data locally for faster future access (default: True)",
    )
    parser.add_argument("--no-cache", dest="cache_benchmark", action="store_false", help="Disable benchmark caching")
    parser.add_argument(
        "--use-cached", action="store_true", default=True, help="Use cached benchmark data if available (default: True)"
    )
    parser.add_argument(
        "--force-download", action="store_true", help="Force fresh download even if cached version exists"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./benchmark_cache",
        help="Directory to store cached benchmark data (default: ./benchmark_cache)",
    )
    parser.add_argument("--cache-status", action="store_true", help="Show cache status and exit")
    parser.add_argument("--cleanup-cache", type=int, metavar="DAYS", help="Clean up cache entries older than DAYS days")

    # Thinking mode control (for Qwen models)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable thinking/reasoning mode (prevents <think> tags for Qwen models)",
    )
