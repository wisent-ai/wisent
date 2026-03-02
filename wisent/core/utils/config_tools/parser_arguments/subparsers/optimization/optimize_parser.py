"""Parser for the 'optimize' command - full model optimization."""

from wisent.core.utils.config_tools.constants import DEFAULT_N_TRIALS


def setup_optimize_parser(parser):
    """
    Set up the optimize command parser.
    
    Usage:
        wisent optimize meta-llama/Llama-3.1-8B-Instruct
        wisent optimize mistralai/Mistral-7B --methods CAA TECZA TETNO GROM
    
    This runs FULL optimization:
    - Classification optimization
    - Steering optimization (ALL methods) for ALL benchmarks + traits
    - Weight modification optimization
    """
    parser.add_argument(
        "model",
        type=str,
        help="Model name or path to optimize (e.g., 'meta-llama/Llama-3.1-8B-Instruct')"
    )
    
    # =========================================================================
    # SCOPE CONTROL
    # =========================================================================
    scope_group = parser.add_argument_group("scope control")
    
    scope_group.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=None,
        help="Specific benchmarks to optimize (default: ALL available benchmarks)"
    )
    scope_group.add_argument(
        "--skip-personalization",
        action="store_true",
        help="Skip personalization trait optimization"
    )
    scope_group.add_argument(
        "--skip-safety",
        action="store_true",
        help="Skip safety/refusal trait optimization"
    )
    scope_group.add_argument(
        "--skip-humanization",
        action="store_true",
        help="Skip humanization trait optimization"
    )
    
    # =========================================================================
    # PHASE CONTROL
    # =========================================================================
    phase_group = parser.add_argument_group("phase control")
    
    phase_group.add_argument(
        "--skip-classification",
        action="store_true",
        help="Skip classification optimization phase"
    )
    phase_group.add_argument(
        "--skip-steering",
        action="store_true",
        help="Skip steering optimization phase"
    )
    phase_group.add_argument(
        "--skip-weights",
        action="store_true",
        help="Skip weight modification optimization phase"
    )
    
    # =========================================================================
    # METHOD SELECTION
    # =========================================================================
    method_group = parser.add_argument_group("steering methods")
    
    method_group.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["CAA", "TECZA", "TETNO", "GROM"],
        choices=["CAA", "TECZA", "TETNO", "GROM"],
        help="Steering methods to test (default: ALL methods)"
    )
    
    # =========================================================================
    # OPTUNA CONFIGURATION
    # =========================================================================
    optuna_group = parser.add_argument_group("Optuna configuration")
    
    optuna_group.add_argument(
        "--n-trials",
        type=int,
        required=True,
        help="Number of Optuna trials per task"
    )
    optuna_group.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_N_TRIALS,
        help="Maximum samples per benchmark (default: 100)"
    )
    
    # =========================================================================
    # EXECUTION OPTIONS
    # =========================================================================
    exec_group = parser.add_argument_group("execution options")
    
    exec_group.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (default: auto-detect)"
    )
    exec_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    exec_group.add_argument(
        "--force",
        action="store_true",
        help="Force re-optimization even if cached results exist (default: skip cached)"
    )
    exec_group.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoint if available (default: True)"
    )
    exec_group.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Start fresh, ignore any existing checkpoint"
    )
