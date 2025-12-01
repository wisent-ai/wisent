"""Parser for the get-activations command."""

import argparse


def setup_get_activations_parser(parser: argparse.ArgumentParser) -> None:
    """
    Set up the get-activations command parser.

    This command loads contrastive pairs from a JSON file, collects activations
    from specified model layers, and saves the enriched pairs back to disk.
    """
    # Input/Output
    parser.add_argument(
        "pairs_file",
        type=str,
        help="Path to JSON file containing contrastive pairs"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path for pairs with activations (JSON)"
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model identifier (e.g., 'meta-llama/Llama-3.2-1B-Instruct')"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cuda, cpu, mps)"
    )

    # Layer selection
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices (e.g., '8,12,15') or 'all' for all layers"
    )

    # Token aggregation
    parser.add_argument(
        "--token-aggregation",
        type=str,
        choices=["average", "final", "first", "max", "min", "max_score"],
        default="average",
        help="How to aggregate token activations. 'max_score' uses highest token hallucination score"
    )

    # Prompt construction strategy
    parser.add_argument(
        "--prompt-strategy",
        type=str,
        choices=["chat_template", "direct_completion", "instruction_following", "multiple_choice", "role_playing"],
        default="chat_template",
        help="Prompt construction strategy (default: chat_template)"
    )

    # Processing options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing (default: 1)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of pairs to process"
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
