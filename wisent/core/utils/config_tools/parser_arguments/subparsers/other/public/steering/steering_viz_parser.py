"""Parser for the 'steering-viz' command - steering effect visualization."""

def setup_steering_viz_parser(parser):
    """Set up the steering-viz command parser."""
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task/benchmark name in database (e.g., truthfulqa_custom)"
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer to visualize"
    )
    parser.add_argument(
        "--strength",
        type=float,
        required=True,
        help="Steering strength multiplier"
    )
    parser.add_argument(
        "--n-test-prompts",
        type=int,
        required=True,
        help="Number of test prompts to run"
    )
    parser.add_argument(
        "--prompt-format",
        type=str,
        required=True,
        choices=["chat", "completion"],
        help="Prompt format"
    )
    parser.add_argument(
        "--extraction-strategy",
        type=str,
        required=True,
        choices=["last_token", "first_token"],
        help="Token extraction strategy"
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="Database URL (default: DATABASE_URL env var)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output PNG file path"
    )
    parser.add_argument(
        "--compare-responses",
        action="store_true",
        help="Also generate and compare text responses (base vs steered)"
    )
    parser.add_argument(
        "--n-response-samples",
        type=int,
        required=True,
        help="Number of samples to show response comparison for"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Max tokens to generate per response (default: 100)"
    )
    parser.add_argument(
        "--per-concept",
        action="store_true",
        help="Generate per-concept steering visualizations with evaluation"
    )
    parser.add_argument(
        "--zwiad-results",
        type=str,
        default=None,
        help="Path to zwiad results JSON (required for --per-concept)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for per-concept visualizations"
    )
    parser.add_argument(
        "--from-cache",
        type=str,
        default=None,
        help="Path to consolidated cache pickle file (alternative to database)"
    )
    parser.add_argument(
        "--space-classifier",
        type=str,
        required=True,
        choices=["logistic", "mlp"],
        help="Classifier for activation space location"
    )
    parser.add_argument(
        "--multipanel",
        action="store_true",
        help="Generate 9-panel visualization (PCA, LDA, t-SNE, UMAP, etc.) like zwiad"
    )
    parser.add_argument(
        "--direction-method",
        type=str,
        required=True,
        choices=["mean_diff", "search", "behavioral", "pca_0"],
        help="Direction discovery method: mean_diff (naive), search (best candidate), behavioral (from labels), pca_0 (first PCA component)"
    )
    parser.add_argument(
        "--steering-method",
        type=str,
        required=True,
        choices=["linear", "clamping", "projection", "replacement", "contrast", "mlp", "adaptive"],
        help="Steering method to use"
    )
    parser.add_argument(
        "--multi-layer",
        action="store_true",
        help="Enable multi-layer steering - steers on ALL layers with per-layer methods"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Optional: restrict multi-layer steering to specific layers (e.g., '8,10,12,14'). If not specified, steers on all layers."
    )
    parser.add_argument(
        "--layer-strengths",
        type=str,
        default=None,
        help="Per-layer strength overrides (e.g., '8:0.5,10:1.0,12:2.0'). Layers not specified use --strength."
    )
    parser.add_argument(
        "--layer-methods",
        type=str,
        default=None,
        help="Per-layer method types (e.g., '8:linear,10:adaptive,12:mlp'). Layers not specified use --steering-method."
    )
    parser.add_argument(
        "--autotune",
        action="store_true",
        help="Automatically tune all parameters (threshold, strength, method) using validation set. Ignores --strength, --steering-method, --layer-strengths, --layer-methods."
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=None,
        help="Fraction of test set to use for validation during autotune (required when --autotune)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Generate interactive HTML visualization with hover tooltips instead of static PNG"
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Device to run model on (e.g., 'cuda', 'mps' for Mac)"
    )
    parser.add_argument(
        "--classifier-test-size",
        type=float,
        required=True,
        help="Fraction of data to hold out for testing in classifier training",
    )
    parser.add_argument(
        "--classifier-batch-size",
        type=int,
        required=True,
        help="Batch size for classifier training",
    )
    parser.add_argument(
        "--classifier-lr",
        type=float,
        required=True,
        help="Learning rate for classifier training",
    )
    parser.add_argument(
        "--mlp-hidden-dim",
        type=int,
        default=None,
        help="Hidden dimension for MLP classifier (required when --space-classifier=mlp)",
    )
    parser.add_argument(
        "--pacmap-neighbors-max", type=int, required=True,
        help="Maximum number of PaCMAP neighbors for visualization",
    )
    parser.add_argument(
        "--pacmap-neighbors-divisor", type=int, required=True,
        help="Divisor for computing PaCMAP neighbors from sample count",
    )
    parser.add_argument(
        "--pacmap-num-iters", type=int, required=True,
        help="Number of PaCMAP iterations for visualization",
    )
