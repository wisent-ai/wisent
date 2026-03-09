"""Steering and advanced arguments for the tasks parser."""

from wisent.core.utils.config_tools.constants import NONSENSE_MAX_WORD_LENGTH


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
        "--steering-strength", type=float, required=True, help="Strength of steering vector application"
    )

    # Steering method selection - uses centralized registry
    from wisent.core.control.steering_methods import SteeringMethodRegistry
    SteeringMethodRegistry.add_all_cli_arguments(parser)

    # Steering output mode selection
    parser.add_argument(
        "--output-mode",
        type=str,
        required=True,
        choices=["likelihoods", "responses", "both"],
        help="Type of comparison to show: 'likelihoods' for log-likelihood comparison only, 'responses' for response generation only, 'both' for both",
    )

    # Token steering arguments
    parser.add_argument("--enable-token-steering", action="store_true", help="Enable token-level steering control")
    parser.add_argument(
        "--token-steering-strategy",
        type=str,
        required=True,
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
        help="Token steering strategy",
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
        required=True,
        help="Maximum steering strength for token strategies",
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
        required=True,
        help="Directory for saving/loading classifiers and vectors",
    )
    # Normalization options
    parser.add_argument("--normalize-mode", action="store_true", help="Enable normalization mode (legacy flag)")
    parser.add_argument(
        "--normalization-method",
        type=str,
        required=True,
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
        required=True,
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
        default=50.0,
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
        "--memory-sampling-interval", type=float, required=True, help="Memory sampling interval in seconds"
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
        required=True,
        choices=["all", "high", "medium", "low"],
        help="Priority level for benchmark selection",
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
    parser.add_argument("--enrichment-max-pairs", type=int, required=True, help="Maximum pairs for enrichment during GROM/TECZA training")
    parser.add_argument("--save-steering-vector", type=str, default=None, help="Path to save the computed steering vector")
    parser.add_argument("--load-steering-vector", type=str, default=None, help="Path to load a pre-computed steering vector")
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
        required=True,
        help="Minimum SNR for early rejection during optimization"
    )
    parser.add_argument(
        "--early-rejection-cv-threshold",
        type=float,
        required=True,
        help="Minimum cross-validation score for early rejection during optimization"
    )

    # Additional output options
    parser.add_argument("--csv-output", type=str, default=None, help="Path to save results in CSV format")
    parser.add_argument("--evaluation-report", type=str, default=None, help="Path to save evaluation report")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue processing other tasks if one fails")

    # Benchmark caching arguments
    parser.add_argument("--cache-benchmark", action="store_true", default=True, help="Cache benchmark data locally (default: True)")
    parser.add_argument("--no-cache", dest="cache_benchmark", action="store_false", help="Disable benchmark caching")
    parser.add_argument("--use-cached", action="store_true", default=True, help="Use cached benchmark data if available")
    parser.add_argument("--force-download", action="store_true", help="Force fresh download even if cached version exists")
    parser.add_argument("--cache-dir", type=str, required=True, help="Directory to store cached benchmark data")
    parser.add_argument("--cache-status", action="store_true", help="Show cache status and exit")
    parser.add_argument("--cleanup-cache", type=int, metavar="DAYS", help="Clean up cache entries older than DAYS days")
    parser.add_argument("--disable-thinking", action="store_true", help="Disable thinking/reasoning mode (prevents <think> tags)")

    # Geometry metric design-choice parameters (non-derivable from data)
    parser.add_argument("--probe-small-hidden", type=int, required=True, dest="probe_small_hidden", help="Hidden dim for small probe")
    parser.add_argument("--probe-mlp-hidden", type=int, required=True, dest="probe_mlp_hidden", help="Hidden dim for MLP probe")
    parser.add_argument("--probe-mlp-alpha", type=float, required=True, dest="probe_mlp_alpha", help="L2 regularization alpha for MLP probe")
    parser.add_argument("--spectral-n-neighbors", type=int, required=True, dest="spectral_n_neighbors", help="Neighbor count for spectral/manifold metrics")
    parser.add_argument("--direction-n-bootstrap", type=int, required=True, dest="direction_n_bootstrap", help="Bootstrap iterations for direction stability")
    parser.add_argument("--direction-subset-fraction", type=float, required=True, dest="direction_subset_fraction", help="Fraction per bootstrap sample")
    parser.add_argument("--direction-std-penalty", type=float, required=True, dest="direction_std_penalty", help="Penalty weight for direction std")
    parser.add_argument("--consistency-w-cosine", type=float, required=True, dest="consistency_w_cosine", help="Weight for cosine in consistency")
    parser.add_argument("--consistency-w-positive", type=float, required=True, dest="consistency_w_positive", help="Weight for positive-fraction in consistency")
    parser.add_argument("--consistency-w-high-sim", type=float, required=True, dest="consistency_w_high_sim", help="Weight for high-similarity in consistency")
    parser.add_argument("--sparsity-threshold-fraction", type=float, required=True, dest="sparsity_threshold_fraction", help="Fraction threshold for sparsity")
    parser.add_argument("--detection-threshold", type=float, required=True, dest="detection_threshold", help="Accuracy threshold for multi-direction detection")
    parser.add_argument("--direction-moderate-similarity", type=float, required=True, dest="direction_moderate_similarity", help="Threshold for moderate similarity")
