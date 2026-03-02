"""Advanced arguments for modify-weights parser."""
import argparse



def setup_advanced_modify_args(parser: argparse.ArgumentParser) -> None:
    """Set up advanced modification arguments."""

    # Additive-specific parameters
    additive_group = parser.add_argument_group("additive parameters")
    additive_group.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Steering strength for additive method"
    )
    additive_group.add_argument(
        "--additive-method",
        type=str,
        default="bias",
        choices=["bias", "weight"],
        help="How to add steering: 'bias' (add to output bias) or 'weight' (modify weight matrix) (default: bias)"
    )

    # Component selection
    parser.add_argument(
        "--components",
        type=str,
        nargs="+",
        default=None,
        help="Components to modify (e.g., 'self_attn.o_proj' 'mlp.down_proj'). Default: method-specific defaults"
    )

    # Extraction component (auto-selects --components if not explicit)
    parser.add_argument(
        "--extraction-component",
        type=str,
        default="residual_stream",
        choices=["residual_stream", "attn_output", "mlp_output", "per_head",
                 "mlp_intermediate", "post_attn_residual", "pre_attn_layernorm",
                 "embedding_output", "final_layernorm", "q_proj", "k_proj",
                 "v_proj", "mlp_gate_activation", "attention_scores", "logits"],
        help=(
            "Extraction component used for steering vectors. Auto-selects "
            "--components if not explicitly set. Default: residual_stream"
        )
    )

    # Kernel-based layer weighting
    kernel_group = parser.add_argument_group("kernel-based layer weighting")
    kernel_group.add_argument(
        "--use-kernel",
        action="store_true",
        help="Use Gaussian-like kernel for smooth layer weighting"
    )
    kernel_group.add_argument(
        "--max-weight",
        type=float,
        required=True,
        help="Peak weight/alpha at center layer"
    )
    kernel_group.add_argument(
        "--max-weight-position",
        type=float,
        default=None,
        help="Layer index for peak weight (default: middle layer)"
    )
    kernel_group.add_argument(
        "--min-weight",
        type=float,
        required=True,
        help="Minimum weight/alpha at edges"
    )
    kernel_group.add_argument(
        "--min-weight-distance",
        type=float,
        default=None,
        help="Distance over which weight decays (default: 60%% of layers)"
    )

    # Vector processing
    parser.add_argument(
        "--normalize-vectors",
        action="store_true",
        help="Normalize steering vectors before modification"
    )

    # Export options
    export_group = parser.add_argument_group("export options")
    export_group.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload modified model to HuggingFace Hub"
    )
    export_group.add_argument(
        "--repo-id",
        type=str,
        help="HuggingFace repository ID (required if --push-to-hub)"
    )
    export_group.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Commit message for Hub upload"
    )
    export_group.add_argument(
        "--private",
        action="store_true",
        help="Make Hub repository private"
    )

    # Optimal config usage
    parser.add_argument(
        "--no-optimal",
        action="store_false",
        dest="use_optimal",
        default=True,
        help="Don't use optimal config from previous optimization (use defaults instead)"
    )

    # Guided modification (linearity-driven) - NOVEL FEATURES
    guided_group = parser.add_argument_group("guided modification (linearity-driven)")
    guided_group.add_argument(
        "--guided",
        action="store_true",
        help="Use linearity-guided weight modification. Automatically selects layers and weights based on measured linear separability."
    )
    guided_group.add_argument(
        "--guided-mode",
        type=str,
        default="adaptive",
        choices=["full", "surgical", "adaptive"],
        help=(
            "Guided ablation mode: "
            "'full' (all layers with signal), "
            "'surgical' (only top-k layers), "
            "'adaptive' (auto-select based on variance). Default: adaptive"
        )
    )
    guided_group.add_argument(
        "--surgical-top-k",
        type=int,
        required=True,
        help="Number of top layers for surgical mode"
    )
    guided_group.add_argument(
        "--min-linear-score",
        type=float,
        required=True,
        help="Minimum linear score to include a layer"
    )
    guided_group.add_argument(
        "--use-fisher-weights",
        action="store_true",
        default=True,
        help="Weight ablation strength by Fisher ratio (default: True)"
    )
    guided_group.add_argument(
        "--no-fisher-weights",
        action="store_true",
        help="Disable Fisher ratio weighting"
    )
    guided_group.add_argument(
        "--extraction-strategy",
        type=str,
        default="chat_last",
        choices=["chat_last", "chat_mean", "chat_max_norm", "completion_last"],
        help="Extraction strategy for guided mode (default: chat_last)"
    )
    
    # Collateral damage validation
    validation_group = parser.add_argument_group("collateral damage validation")
    validation_group.add_argument(
        "--validate-collateral",
        action="store_true",
        help="Validate that modification doesn't hurt unrelated representations"
    )
    validation_group.add_argument(
        "--max-degradation",
        type=float,
        required=True,
        help="Maximum allowed degradation on validation benchmarks"
    )
    validation_group.add_argument(
        "--validation-benchmarks",
        type=str,
        nargs="+",
        default=None,
        help="Benchmarks to use for collateral damage validation"
    )
    
    # Multi-concept modification
    multi_concept_group = parser.add_argument_group("multi-concept modification")
    multi_concept_group.add_argument(
        "--concepts",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Multiple concepts to modify simultaneously. Format: 'name:action:strength' "
            "e.g., 'refusal:suppress:1.0' 'truthfulness:enhance:0.5'. "
            "Actions: suppress, enhance"
        )
    )
    multi_concept_group.add_argument(
        "--orthogonalize-concepts",
        action="store_true",
        default=True,
        help="Orthogonalize concept directions to minimize interference (default: True)"
    )
    multi_concept_group.add_argument(
        "--no-orthogonalize",
        action="store_true",
        help="Disable concept orthogonalization"
    )

    # General options
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
    parser.add_argument(
        "--save-steering-vectors",
        type=str,
        default=None,
        help="Save generated steering vectors to file (optional)"
    )
    parser.add_argument(
        "--save-diagnostics",
        type=str,
        default=None,
        help="Save layer diagnostics to JSON file (for guided mode)"
    )
