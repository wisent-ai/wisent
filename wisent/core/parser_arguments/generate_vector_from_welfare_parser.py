"""Parser for the generate-vector-from-welfare command."""

import argparse

# Available welfare traits (ANIMA framework)
WELFARE_TRAITS = [
    "comfort_distress",
    "satisfaction_dissatisfaction",
    "engagement_aversion",
    "curiosity_boredom",
    "affiliation_isolation",
    "agency_helplessness",
]


def setup_generate_vector_from_welfare_parser(parser: argparse.ArgumentParser) -> None:
    """
    Set up the generate-vector-from-welfare command parser.

    This command runs the complete pipeline using pre-loaded welfare pairs:
    1. Load pre-generated welfare pairs from storage (ANIMA framework)
    2. Collect activations from those pairs
    3. Create steering vectors from the activations

    Available welfare traits (based on ANIMA framework):
    - comfort_distress: Ease vs suffering in interactions
    - satisfaction_dissatisfaction: Fulfillment vs disappointment
    - engagement_aversion: Approach vs withdrawal tendencies
    - curiosity_boredom: Interest vs disengagement
    - affiliation_isolation: Connection vs loneliness
    - agency_helplessness: Control vs powerlessness
    """
    # Welfare trait to use
    parser.add_argument(
        "--trait",
        type=str,
        required=True,
        choices=WELFARE_TRAITS,
        help="Welfare trait to generate steering vector for"
    )

    # Direction (positive or negative pole of the trait)
    parser.add_argument(
        "--direction",
        type=str,
        default="positive",
        choices=["positive", "negative"],
        help="Direction to steer: 'positive' (comfort, satisfaction, etc.) "
             "or 'negative' (distress, dissatisfaction, etc.) (default: positive)"
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path for the final steering vector (.pt or .json)"
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (e.g., 'auto', 'cpu', 'cuda', 'cuda:0', 'mps')"
    )

    # Pair selection
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=None,
        help="Number of pairs to use (default: all available, typically 100)"
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
        choices=[
            "chat_mean", "chat_first", "chat_last", "chat_gen_point",
            "chat_max_norm", "chat_weighted", "role_play", "mc_balanced",
            "completion_last", "completion_mean", "mc_completion"
        ],
        default="chat_last",
        help="Extraction strategy for activations (default: chat_last)"
    )

    # Steering vector creation
    parser.add_argument(
        "--method",
        type=str,
        choices=["caa", "hyperplane", "mlp", "prism", "pulse", "titan"],
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

    # Quality control
    parser.add_argument(
        "--accept-low-quality-vector",
        action="store_true",
        default=False,
        help="Accept steering vectors that fail quality checks (convergence, SNR, etc.)"
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
