"""Parser for the generate-vector-from-task command."""

import argparse


def setup_generate_vector_from_task_parser(parser: argparse.ArgumentParser) -> None:
    """
    Set up the generate-vector-from-task command parser.
    
    This command runs the complete pipeline:
    1. Generate contrastive pairs from an lm-eval task
    2. Collect activations from those pairs
    3. Create steering vectors from the activations
    
    All in one command.
    """
    # Task source
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Name of the lm-eval task to use (e.g., 'mmlu', 'hellaswag')"
    )
    parser.add_argument(
        "--trait-label",
        type=str,
        required=True,
        help="Label for the trait being steered (e.g., 'accuracy', 'correctness')"
    )
    
    # Output
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path for the final steering vector (JSON)"
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace model name or path (default: meta-llama/Llama-3.2-1B-Instruct)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (e.g., 'cpu', 'cuda', 'cuda:0')"
    )
    
    # Pair generation
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=50,
        help="Number of contrastive pairs to generate (default: 50)"
    )
    
    # Activation collection
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices (e.g., '8,12,16') or 'all' (default: all layers)"
    )
    parser.add_argument(
        "--token-aggregation",
        type=str,
        choices=["average", "final", "first", "max", "continuation"],
        default="average",
        help="How to aggregate token activations (default: average)"
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
        help="L2-normalize steering vectors (default: True)"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_false",
        dest="normalize",
        help="Do not L2-normalize steering vectors"
    )
    
    # Intermediate file handling
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep intermediate files (pairs and enriched pairs)"
    )
    parser.add_argument(
        "--intermediate-dir",
        type=str,
        default=None,
        help="Directory for intermediate files (default: same as output)"
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
