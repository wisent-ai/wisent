"""Parser for the 'steering-viz' command - steering effect visualization."""


def setup_steering_viz_parser(parser):
    """Set up the steering-viz command parser."""
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model name (default: meta-llama/Llama-3.2-1B-Instruct)"
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
        default=12,
        help="Layer to visualize (default: 12)"
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Steering strength multiplier (default: 1.0)"
    )
    parser.add_argument(
        "--n-test-prompts",
        type=int,
        default=50,
        help="Number of test prompts to run (default: 50)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum reference pairs to load (default: 200)"
    )
    parser.add_argument(
        "--prompt-format",
        type=str,
        default="chat",
        choices=["chat", "completion"],
        help="Prompt format (default: chat)"
    )
    parser.add_argument(
        "--extraction-strategy",
        type=str,
        default="last_token",
        choices=["last_token", "first_token"],
        help="Token extraction strategy (default: last_token)"
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
        default="./steering_effect.png",
        help="Output PNG file path (default: ./steering_effect.png)"
    )
    parser.add_argument(
        "--compare-responses",
        action="store_true",
        help="Also generate and compare text responses (base vs steered)"
    )
    parser.add_argument(
        "--n-response-samples",
        type=int,
        default=5,
        help="Number of samples to show response comparison for (default: 5)"
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
        "--repscan-results",
        type=str,
        default=None,
        help="Path to repscan results JSON (required for --per-concept)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./concept_steering_viz",
        help="Output directory for per-concept visualizations (default: ./concept_steering_viz)"
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
        default="mlp",
        choices=["logistic", "mlp"],
        help="Classifier for activation space location (default: mlp)"
    )
    parser.add_argument(
        "--multipanel",
        action="store_true",
        help="Generate 9-panel visualization (PCA, LDA, t-SNE, UMAP, etc.) like repscan"
    )
    parser.add_argument(
        "--direction-method",
        type=str,
        default="mean_diff",
        choices=["mean_diff", "search", "behavioral", "pca_0"],
        help="Direction discovery method: mean_diff (naive), search (best candidate), behavioral (from labels), pca_0 (first PCA component)"
    )
    parser.add_argument(
        "--steering-method",
        type=str,
        default="linear",
        choices=["linear", "clamping", "projection", "replacement", "contrast", "mlp", "adaptive"],
        help="Steering method to use (default: linear)"
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
        default=0.5,
        help="Fraction of test set to use for validation during autotune (default: 0.5)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Generate interactive HTML visualization with hover tooltips instead of static PNG"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run model on (default: cuda, can be 'mps' for Mac)"
    )
