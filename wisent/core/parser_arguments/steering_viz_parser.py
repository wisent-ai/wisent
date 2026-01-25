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
