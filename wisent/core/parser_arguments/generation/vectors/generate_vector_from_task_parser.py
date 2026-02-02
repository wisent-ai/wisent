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
        default="auto",
        help="Device to use (e.g., 'auto', 'cpu', 'cuda', 'cuda:0', 'mps')"
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
        "--extraction-strategy",
        type=str,
        choices=["chat_mean", "chat_first", "chat_last", "chat_gen_point", "chat_max_norm", "chat_weighted", "role_play", "mc_balanced", "completion_last", "completion_mean", "mc_completion"],
        default="chat_mean",
        help="Extraction strategy. Chat models: chat_mean, chat_first, chat_last, chat_max_norm, chat_weighted, role_play, mc_balanced. Base models: completion_last, completion_mean, mc_completion"
    )
    
    # Steering vector creation
    parser.add_argument(
        "--method",
        type=str,
        choices=["caa", "prism", "pulse", "titan"],
        default="caa",
        help="Steering method to use (default: caa). If optimal config exists, method is auto-selected."
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
    
    # Universal Subspace options (PRISM/TITAN)
    parser.add_argument(
        "--auto-num-directions",
        action="store_true",
        default=False,
        help="Automatically determine num_directions based on explained variance (PRISM/TITAN)"
    )
    parser.add_argument(
        "--use-universal-basis-init",
        action="store_true",
        default=False,
        help="Initialize directions from universal basis (PRISM/TITAN)"
    )
    parser.add_argument(
        "--num-directions",
        type=int,
        default=3,
        help="Number of steering directions for PRISM/TITAN (default: 3)"
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
    
    # Quality control
    parser.add_argument(
        "--accept-low-quality-vector",
        action="store_true",
        default=False,
        help="Accept steering vectors that fail quality checks (convergence, SNR, etc.)"
    )
    
    # Optimal config usage
    parser.add_argument(
        "--no-optimal",
        action="store_false",
        dest="use_optimal",
        default=True,
        help="Don't use optimal config from previous optimization (use defaults instead)"
    )
    parser.add_argument(
        "--show-optimal",
        action="store_true",
        help="Show optimal config if available, but don't apply it"
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
