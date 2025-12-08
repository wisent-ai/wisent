"""Parser for the simple 'optimize' command."""


def setup_optimize_parser(parser):
    """
    Set up the optimize command parser.
    
    Usage:
        wisent optimize meta-llama/Llama-3.1-8B-Instruct
        wisent optimize Qwen/Qwen2-7B --quick
        wisent optimize mistralai/Mistral-7B --methods CAA PRISM
    """
    parser.add_argument(
        "model",
        type=str,
        help="Model name or path to optimize (e.g., 'meta-llama/Llama-3.1-8B-Instruct')"
    )
    
    # Benchmark selection
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=None,
        help="Specific benchmarks to optimize (default: core benchmarks)"
    )
    parser.add_argument(
        "--only",
        type=str,
        nargs="+",
        default=None,
        help="Only optimize these specific benchmarks from the default set"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: optimize only 4 key benchmarks (truthfulqa, arc_easy, hellaswag, gsm8k)"
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Extended mode: optimize additional benchmarks beyond core set"
    )
    
    # Method selection
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["CAA"],
        choices=["CAA", "PRISM", "PULSE", "TITAN"],
        help="Steering methods to test (default: CAA)"
    )
    
    # Search configuration
    parser.add_argument(
        "--quick-search",
        action="store_true",
        help="Use reduced search space for faster optimization"
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=50,
        help="Maximum configurations to test per method per benchmark (default: 50)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum samples per benchmark (default: 100)"
    )
    
    # Execution options
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-optimize benchmarks even if already optimized"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (default: auto-detect)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
