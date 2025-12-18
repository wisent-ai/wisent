"""
Parser setup for the 'tasks' command.

This command runs evaluation tasks on language models.
"""


def setup_tasks_parser(parser):
    """Set up the tasks subcommand parser."""

    # Task listing options (mutually exclusive with task execution)
    list_group = parser.add_mutually_exclusive_group()
    list_group.add_argument(
        "--list-tasks",
        action="store_true",
        help="List all 37 available benchmark tasks organized by priority (excludes 28 known problematic benchmarks)",
    )
    list_group.add_argument(
        "--task-info", type=str, metavar="TASK_NAME", help="Show detailed information about a specific task"
    )
    list_group.add_argument("--all", action="store_true", help="Run all 37 available benchmarks automatically")

    # Task execution argument (optional when using listing commands or --all)
    parser.add_argument(
        "task_names",
        nargs="?",
        help="Comma-separated list of available task names (37 working benchmarks), or path to CSV/JSON file with --from-csv/--from-json (not needed with --all)",
    )

    # Skills/risks based task selection
    parser.add_argument(
        "--skills", type=str, nargs="+", help="Select tasks by skill categories (e.g., coding, mathematics, reasoning)"
    )
    parser.add_argument(
        "--risks",
        type=str,
        nargs="+",
        help="Select tasks by risk categories (e.g., harmfulness, toxicity, hallucination)",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Number of tasks to randomly select from matched tasks (default: all)",
    )
    parser.add_argument(
        "--min-quality-score",
        type=int,
        default=2,
        choices=[1, 2, 3, 4, 5],
        help="Minimum quality score for tasks when using --skills/--risks (default: 2)",
    )
    parser.add_argument(
        "--task-seed", type=int, default=None, help="Random seed for task selection (for reproducibility)"
    )

    # Mixed sampling from multiple benchmarks
    parser.add_argument(
        "--tag",
        type=str,
        nargs="+",
        help="Sample randomly from all benchmarks with these tags (e.g., --tag coding). Creates a mixed dataset from multiple benchmarks.",
    )
    parser.add_argument(
        "--mixed-samples",
        type=int,
        default=1000,
        help="Total number of samples to collect when using --tag (default: 1000)",
    )
    parser.add_argument(
        "--tag-mode",
        type=str,
        choices=["any", "all"],
        default="any",
        help="Whether benchmarks must have ANY or ALL specified tags (default: any)",
    )

    # Cross-benchmark evaluation
    parser.add_argument(
        "--train-task", type=str, help="Task/benchmark to train on (can be a task name or --tag for mixed)"
    )
    parser.add_argument(
        "--eval-task", type=str, help="Task/benchmark to evaluate on (can be a task name or --tag for mixed)"
    )
    parser.add_argument(
        "--train-tag", type=str, nargs="+", help="Tags for training data when using cross-benchmark evaluation"
    )
    parser.add_argument(
        "--eval-tag", type=str, nargs="+", help="Tags for evaluation data when using cross-benchmark evaluation"
    )
    parser.add_argument(
        "--cross-benchmark",
        action="store_true",
        help="Enable cross-benchmark evaluation mode (train on one, eval on another)",
    )

    # Synthetic pair generation
    parser.add_argument(
        "--synthetic", action="store_true", help="Generate synthetic contrastive pairs from a trait description"
    )
    parser.add_argument(
        "--trait",
        type=str,
        help="Natural language description of desired model behavior (e.g., 'hallucinates less', 'more factual', 'less verbose')",
    )
    parser.add_argument(
        "--num-synthetic-pairs", type=int, default=30, help="Number of synthetic pairs to generate (default: 30)"
    )
    parser.add_argument("--save-synthetic", type=str, help="Path to save generated synthetic pairs as JSON")
    parser.add_argument(
        "--load-synthetic", type=str, help="Path to load previously generated synthetic pairs from JSON"
    )

    # Nonsense pair generation (special case of synthetic)
    parser.add_argument(
        "--nonsense",
        action="store_true",
        help="Generate nonsense contrastive pairs (negative responses are gibberish/nonsense). Automatically enables --synthetic mode.",
    )
    parser.add_argument(
        "--nonsense-mode",
        type=str,
        choices=["random_chars", "repetitive", "word_salad", "mixed"],
        default="random_chars",
        help="Type of nonsense to generate: 'random_chars' (ahsdhashdahsdha), 'repetitive' (the the the), 'word_salad' (real words, no meaning), 'mixed' (combination). Default: random_chars",
    )

    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or path")
    parser.add_argument(
        "--layer",
        type=str,
        default="15",
        help="Layer(s) to extract activations from. Can be a single layer (15), range (14-16), or comma-separated list (14,15,16)",
    )
    parser.add_argument("--shots", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Train/test split ratio")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents per task")
    parser.add_argument(
        "--training-limit",
        type=int,
        default=None,
        help="Limit number of training documents (overrides limit for training)",
    )
    parser.add_argument(
        "--testing-limit",
        type=int,
        default=None,
        help="Limit number of testing documents (overrides limit for testing)",
    )
    parser.add_argument("--output", type=str, default="./results", help="Output directory for results")
    parser.add_argument(
        "--classifier-type", type=str, choices=["logistic", "mlp"], default="logistic", help="Type of classifier"
    )
    parser.add_argument("--max-new-tokens", type=int, default=300, help="Maximum new tokens for generation")
    parser.add_argument("--device", type=str, default=None, help="Device to run on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    # Extraction strategy - unified approach combining prompt format and token selection
    from wisent.core.activations.extraction_strategy import ExtractionStrategy
    parser.add_argument(
        "--extraction-strategy",
        type=str,
        choices=ExtractionStrategy.list_all(),
        default=ExtractionStrategy.default().value,
        help=f"Extraction strategy for activations. Options: {', '.join(ExtractionStrategy.list_all())}. Default: {ExtractionStrategy.default().value}",
    )
    parser.add_argument(
        "--ground-truth-method",
        type=str,
        choices=[
            "none",
            "exact_match",
            "substring_match",
            "user_specified",
            "interactive",
            "manual_review",
            "good",
            "lm-eval-harness",
        ],
        default="lm-eval-harness",
        help="Method for ground truth evaluation. 'lm-eval-harness' uses lm-eval-harness tasks for evaluation (default for most tasks), 'none' skips evaluation, 'exact_match' and 'substring_match' are problematic for free-form generation, 'user_specified' allows manual labeling, 'interactive' prompts for y/n labeling, 'manual_review' marks for review, 'good' marks everything as truthful (for debugging)",
    )
    parser.add_argument(
        "--user-labels",
        type=str,
        nargs="*",
        default=None,
        help="User-specified ground truth labels for responses ('truthful' or 'hallucination'). Used with --ground-truth-method user_specified",
    )

    # File input arguments
    parser.add_argument(
        "--from-csv",
        action="store_true",
        help="Load task data from CSV file. Requires columns: question, correct_answer, incorrect_answer",
    )
    parser.add_argument(
        "--from-json",
        action="store_true",
        help="Load task data from JSON file. Expected format: list of objects with question, correct_answer, incorrect_answer",
    )
    parser.add_argument(
        "--question-col", type=str, default="question", help="Column name for questions in CSV file (default: question)"
    )
    parser.add_argument(
        "--correct-col",
        type=str,
        default="correct_answer",
        help="Column name for correct answers in CSV file (default: correct_answer)",
    )
    parser.add_argument(
        "--incorrect-col",
        type=str,
        default="incorrect_answer",
        help="Column name for incorrect answers in CSV file (default: incorrect_answer)",
    )

    # Optimization arguments
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable hyperparameter optimization. When enabled, will find optimal layer, threshold, and aggregation method",
    )
    parser.add_argument(
        "--optimize-layers",
        type=str,
        default="all",
        help="Layer range for optimization (e.g., '8-24' or '10,15,20' or 'all'). Default: all (uses all model layers)",
    )
    parser.add_argument(
        "--optimize-metric",
        type=str,
        choices=["accuracy", "f1", "precision", "recall", "auc"],
        default="f1",
        help="Metric to optimize for. Default: f1",
    )
    parser.add_argument(
        "--optimize-max-combinations",
        type=int,
        default=100,
        help="Maximum number of hyperparameter combinations to test. Default: 100",
    )
    parser.add_argument(
        "--auto-optimize",
        action="store_true",
        help="Automatically enable optimization when layer is not specified or is -1",
    )

    # Dataset validation arguments
    parser.add_argument(
        "--allow-small-dataset",
        action="store_true",
        help="Allow training with datasets smaller than 4 samples (may cause training issues)",
    )

    # Detection handling arguments
    parser.add_argument(
        "--detection-action",
        type=str,
        choices=["pass_through", "replace_with_placeholder", "regenerate_until_safe"],
        default="pass_through",
        help="Action to take when problematic content is detected (default: pass_through)",
    )
    parser.add_argument(
        "--placeholder-message",
        type=str,
        default=None,
        help="Custom placeholder message for detected content (if not specified, uses default)",
    )
    parser.add_argument(
        "--max-regeneration-attempts",
        type=int,
        default=3,
        help="Maximum attempts to regenerate safe content (default: 3)",
    )
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=0.6,
        help="Threshold for classification (higher = more strict detection) (default: 0.6)",
    )
    parser.add_argument("--log-detections", action="store_true", help="Enable logging of detection events")

    # Code execution security arguments
    parser.add_argument(
        "--trust-code-execution",
        action="store_true",
        help="⚠️  UNSAFE: Allow code execution without Docker in trusted sandbox environments (e.g., RunPod containers). Use only in secure, isolated environments!",
    )

    # Steering mode arguments
    parser.add_argument(
        "--steering-mode", action="store_true", help="Enable steering mode (uses CAA vectors instead of classification)"
    )
    parser.add_argument(
        "--steering-strength", type=float, default=1.0, help="Strength of steering vector application (default: 1.0)"
    )

    # Steering method selection - uses centralized registry
    from wisent.core.steering_methods import SteeringMethodRegistry
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
        default=0.5,
        help="Decay rate for exponential token steering strategies (0-1, default: 0.5)",
    )
    parser.add_argument(
        "--token-min-strength",
        type=float,
        default=0.1,
        help="Minimum steering strength for token strategies (default: 0.1)",
    )
    parser.add_argument(
        "--token-max-strength",
        type=float,
        default=1.0,
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
        default=0.1,
        help="Strength multiplier for prompt tokens (default: 0.1)",
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
        default=20,
        help="Maximum reasonable word length for nonsense detection (default: 20)",
    )
    parser.add_argument(
        "--repetition-threshold",
        type=float,
        default=0.7,
        help="Threshold for repetitive content detection (0-1, default: 0.7)",
    )
    parser.add_argument(
        "--gibberish-threshold",
        type=float,
        default=0.3,
        help="Threshold for gibberish word detection (0-1, default: 0.3)",
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
        "--memory-sampling-interval", type=float, default=0.1, help="Memory sampling interval in seconds (default: 0.1)"
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
        default=5.0,
        help="Minimum SNR for early rejection during optimization (default: 5.0)"
    )
    parser.add_argument(
        "--early-rejection-cv-threshold",
        type=float,
        default=0.1,
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
