"""Parser for the get-activations command."""

import argparse

from wisent.core.utils.config_tools.constants.for_experiments.by_domain.analysis._analysis import (
    CHAT_WEIGHTED_DECAY_DEFAULT,
)


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
        required=True,
        help="Model identifier (HuggingFace model name or path)"
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

    # Extraction strategy (combines prompt format and token selection)
    parser.add_argument(
        "--extraction-strategy",
        type=str,
        choices=["chat_mean", "chat_first", "chat_last", "chat_gen_point", "chat_max_norm", "chat_weighted", "role_play", "mc_balanced", "completion_last", "completion_mean", "mc_completion"],
        required=True,
        help="Extraction strategy. Chat models: chat_mean, chat_first, chat_last, chat_max_norm, chat_weighted, role_play, mc_balanced. Base models: completion_last, completion_mean, mc_completion"
    )

    parser.add_argument(
        "--extraction-component",
        type=str,
        required=True,
        choices=["residual_stream", "attn_output", "mlp_output", "per_head",
                 "mlp_intermediate", "post_attn_residual", "pre_attn_layernorm",
                 "embedding_output", "final_layernorm", "q_proj", "k_proj",
                 "v_proj", "mlp_gate_activation", "attention_scores", "logits"],
        help="Transformer component to extract from"
    )

    # Raw mode - output full hidden states instead of extracted vectors
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Output raw hidden states [seq_len, hidden_size] instead of extracted vectors. "
             "Allows applying different extraction strategies later without re-running the model."
    )

    # Strategy-specific parameters
    parser.add_argument(
        "--weighted-decay",
        type=float,
        default=CHAT_WEIGHTED_DECAY_DEFAULT,
        help="Exponential decay factor for chat_weighted strategy. Higher values "
             "weight earlier answer tokens more heavily; 0.0 reduces to chat_mean. "
             "Used only when --extraction-strategy=chat_weighted."
    )

    # Processing options
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Batch size for processing"
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
