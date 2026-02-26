"""Basic arguments for the tasks parser."""

from wisent.core.constants import DEFAULT_SPLIT_RATIO, DEFAULT_RANDOM_SEED, DEFAULT_LAYER, PARSER_DEFAULT_NUM_PAIRS_GENERATE, OPTIMIZE_MAX_COMBINATIONS_DEFAULT, PARSER_DEFAULT_MIN_QUALITY, PARSER_DEFAULT_TOTAL_SAMPLES, PARSER_DEFAULT_FEW_SHOT, PARSER_DEFAULT_MAX_REGEN_ATTEMPTS, PARSER_DEFAULT_CLASSIFICATION_THRESHOLD


def setup_basic_task_args(parser):
    """Set up basic task arguments: listing, execution, extraction, dataset.
    
    Called from setup_tasks_parser.
    """
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
        default=PARSER_DEFAULT_MIN_QUALITY,
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
        default=PARSER_DEFAULT_TOTAL_SAMPLES,
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
        "--num-synthetic-pairs", type=int, default=PARSER_DEFAULT_NUM_PAIRS_GENERATE, help="Number of synthetic pairs to generate (default: 30)"
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
        default=str(DEFAULT_LAYER),
        help="Layer(s) to extract activations from. Can be a single layer, range (14-16), or comma-separated list (14,15,16)",
    )
    parser.add_argument("--shots", type=int, default=PARSER_DEFAULT_FEW_SHOT, help="Number of few-shot examples")
    parser.add_argument("--split-ratio", type=float, default=DEFAULT_SPLIT_RATIO, help="Train/test split ratio")
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
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    # Extraction strategy - unified approach combining prompt format and token selection
    from wisent.core.activations import ExtractionStrategy
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
        default=OPTIMIZE_MAX_COMBINATIONS_DEFAULT,
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
        default=PARSER_DEFAULT_MAX_REGEN_ATTEMPTS,
        help="Maximum attempts to regenerate safe content (default: 3)",
    )
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=PARSER_DEFAULT_CLASSIFICATION_THRESHOLD,
        help="Threshold for classification (higher = more strict detection) (default: 0.6)",
    )
    parser.add_argument("--log-detections", action="store_true", help="Enable logging of detection events")

    # Code execution security arguments
    parser.add_argument(
        "--trust-code-execution",
        action="store_true",
        help="⚠️  UNSAFE: Allow code execution without Docker in trusted sandbox environments (e.g., RunPod containers). Use only in secure, isolated environments!",
    )

