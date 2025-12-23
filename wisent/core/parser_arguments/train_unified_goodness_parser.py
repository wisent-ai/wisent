"""Parser for the train-unified-goodness command."""

import argparse


def setup_train_unified_goodness_parser(parser: argparse.ArgumentParser) -> None:
    """
    Set up the train-unified-goodness command parser.

    This command:
    1. Pools contrastive pairs from ALL supported benchmarks
    2. Trains a single unified "goodness" steering vector from pooled data
    3. Evaluates the vector across ALL benchmarks (pooled evaluation)

    The result is a single direction that captures "correctness" across all tasks.
    """
    # Output
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path for the unified steering vector (.pt)"
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="JunHowie/Qwen3-8B-GPTQ-Int4",
        help="HuggingFace model name or path (default: JunHowie/Qwen3-8B-GPTQ-Int4)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (e.g., 'auto', 'cpu', 'cuda', 'cuda:0', 'mps')"
    )

    # Benchmark selection
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help=(
            "Task(s) to train on. Can be: "
            "single benchmark (e.g., 'arc_easy'), "
            "comma-separated benchmarks (e.g., 'arc_easy,gsm8k,hellaswag'), "
            "or omit to use ALL benchmarks"
        )
    )
    parser.add_argument(
        "--exclude-benchmarks",
        type=str,
        nargs="+",
        default=None,
        help="Benchmarks to exclude from training/evaluation"
    )
    parser.add_argument(
        "--max-benchmarks",
        type=int,
        default=None,
        help="Maximum number of benchmarks to include (for faster testing)"
    )

    # Data sampling
    parser.add_argument(
        "--cap-pairs-per-benchmark",
        type=int,
        default=None,
        help="Cap pairs per benchmark at this number. Benchmarks with more pairs will be randomly sampled down. "
             "Example: --cap-pairs-per-benchmark 10000 means any benchmark with >10k pairs gets 10k random samples"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of pairs used for training vs evaluation (default: 0.8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    # Activation collection
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Specific layer to use. If not set, uses middle layer of model"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Multiple layers as comma-separated indices (e.g., '12,16,20') or range ('10-20')"
    )
    parser.add_argument(
        "--token-aggregation",
        type=str,
        choices=["average", "final", "first", "max", "continuation"],
        default="continuation",
        help="How to aggregate token activations (default: continuation)"
    )
    parser.add_argument(
        "--prompt-strategy",
        type=str,
        choices=["chat_template", "direct_completion", "instruction_following", "multiple_choice", "role_playing"],
        default="chat_template",
        help="Prompt construction strategy (default: chat_template)"
    )

    # Steering vector creation
    parser.add_argument(
        "--method",
        type=str,
        choices=["caa"],
        default="caa",
        help="Steering method to use (default: caa)"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="L2-normalize steering vector (default: True)"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_false",
        dest="normalize",
        help="Do not L2-normalize steering vector"
    )

    # Evaluation options
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation after training (just produce the vector)"
    )
    parser.add_argument(
        "--evaluate-steering-scales",
        type=str,
        default="0.0,0.5,1.0,1.5,2.0",
        help="Comma-separated steering scales to evaluate (default: 0.0,0.5,1.0,1.5,2.0)"
    )

    # Output options
    parser.add_argument(
        "--save-pairs",
        type=str,
        default=None,
        help="Save pooled pairs to this file (for debugging/inspection)"
    )
    parser.add_argument(
        "--save-report",
        type=str,
        default=None,
        help="Save evaluation report to this JSON file"
    )

    # Display options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Show timing information"
    )
