"""Parser for the 'zwiad' command - geometry analysis with concept decomposition."""


def setup_zwiad_parser(parser):
    """
    Set up the zwiad command parser.

    Usage:
        wisent zwiad --from-database --model meta-llama/Llama-3.2-1B-Instruct --task truthfulqa_custom
        wisent zwiad --from-cache cache/optimize/meta-llama_Llama-3.2-1B-Instruct/activations_xxx.pkl
        wisent zwiad --from-database --task truthfulqa_custom --layers 8-16 --visualizations
    """
    # Data source - mutually exclusive
    data_source = parser.add_mutually_exclusive_group(required=True)
    data_source.add_argument(
        "--from-database",
        action="store_true",
        help="Load activations from Supabase database"
    )
    data_source.add_argument(
        "--from-cache",
        type=str,
        metavar="PATH",
        help="Load activations from local cache pickle file"
    )

    # Model and task (required only for --from-database)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name in database"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task/benchmark name in database (required for --from-database, optional for --from-cache to enable LLM concept naming)"
    )

    # Layer selection
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Layers to analyze: range (8-16) or comma-separated (8,10,12). Default: all available layers in database"
    )

    # Database options
    parser.add_argument(
        "--prompt-format",
        type=str,
        required=True,
        choices=["chat", "completion"],
        help="Prompt format used for activations"
    )
    parser.add_argument(
        "--extraction-strategy",
        type=str,
        required=True,
        choices=["last_token", "first_token", "chat_last"],
        help="Token extraction strategy"
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
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of pairs to load (default: all)"
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="Database URL (default: DATABASE_URL env var)"
    )

    # Protocol steps
    parser.add_argument(
        "--steps",
        type=str,
        required=True,
        help="Protocol steps to run: 'all', 'signal', 'geometry', 'decomposition', 'intervention', 'editability', or comma-separated (e.g., 'signal,geometry,editability')"
    )

    # Analysis options
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        default=False,
        help="Disable visualization generation (visualizations are on by default)"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        required=True,
        help="LLM model for concept naming"
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for JSON results"
    )
    parser.add_argument(
        "--visualizations-dir",
        type=str,
        required=True,
        help="Directory to save visualization PNGs"
    )
