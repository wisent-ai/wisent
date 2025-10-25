"""Parser setup for the 'agent' command."""


def setup_agent_parser(parser):
    """Set up the agent subcommand parser."""
    parser.add_argument("prompt", type=str, help="Prompt to send to the autonomous agent")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model to use")
    parser.add_argument("--layer", type=int, help="Layer to use (overrides parameter file)")
    parser.add_argument(
        "--quality-threshold", type=float, default=0.3, help="Quality threshold for classifiers (default: 0.3)"
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=10.0,
        help="Time budget in minutes for creating classifiers (default: 10.0)",
    )
    parser.add_argument("--max-attempts", type=int, default=3, help="Maximum improvement attempts (default: 3)")
    parser.add_argument(
        "--max-classifiers", type=int, default=None, help="Maximum classifiers to use (default: no limit)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # Steering method arguments
    parser.add_argument(
        "--steering-method",
        type=str,
        default="CAA",
        choices=["CAA", "HPR", "DAC", "BiPO", "KSteering"],
        help="Steering method to use (default: CAA)",
    )
    parser.add_argument(
        "--steering-strength", type=float, default=1.0, help="Strength of steering vector application (default: 1.0)"
    )
    parser.add_argument("--steering-mode", action="store_true", help="Enable steering mode")

    # Normalization parameters
    parser.add_argument("--normalize-mode", action="store_true", help="Enable normalization of steering vectors")
    parser.add_argument(
        "--normalization-method",
        type=str,
        default="none",
        choices=["none", "l2_unit", "l2_norm", "max_norm"],
        help="Normalization method for steering vectors (default: none)",
    )
    parser.add_argument("--target-norm", type=float, default=None, help="Target norm for steering vectors")

    # HPR (Householder Pseudo-Rotation) parameters
    parser.add_argument("--hpr-beta", type=float, default=1.0, help="Beta parameter for HPR steering (default: 1.0)")

    # DAC (Dynamic Activation Composition) parameters
    parser.add_argument("--dac-dynamic-control", action="store_true", help="Enable dynamic control for DAC steering")
    parser.add_argument(
        "--dac-entropy-threshold", type=float, default=1.0, help="Entropy threshold for DAC steering (default: 1.0)"
    )

    # BiPO (Bi-directional Preference Optimization) parameters
    parser.add_argument("--bipo-beta", type=float, default=0.1, help="Beta parameter for BiPO steering (default: 0.1)")
    parser.add_argument(
        "--bipo-learning-rate", type=float, default=5e-4, help="Learning rate for BiPO steering (default: 5e-4)"
    )
    parser.add_argument(
        "--bipo-epochs", type=int, default=100, help="Number of epochs for BiPO steering (default: 100)"
    )

    # KSteering parameters
    parser.add_argument(
        "--ksteering-num-labels", type=int, default=6, help="Number of labels for K-steering (default: 6)"
    )
    parser.add_argument(
        "--ksteering-hidden-dim", type=int, default=512, help="Hidden dimension for K-steering (default: 512)"
    )
    parser.add_argument(
        "--ksteering-learning-rate", type=float, default=1e-3, help="Learning rate for K-steering (default: 1e-3)"
    )
    parser.add_argument(
        "--ksteering-classifier-epochs", type=int, default=100, help="Classifier epochs for K-steering (default: 100)"
    )
    parser.add_argument(
        "--ksteering-target-labels",
        type=str,
        default="0",
        help="Target labels for K-steering (comma-separated, default: '0')",
    )
    parser.add_argument(
        "--ksteering-avoid-labels",
        type=str,
        default="",
        help="Avoid labels for K-steering (comma-separated, default: '')",
    )
    parser.add_argument(
        "--ksteering-alpha", type=float, default=50.0, help="Alpha parameter for K-steering (default: 50.0)"
    )

    # Quality Control System parameters
    parser.add_argument(
        "--enable-quality-control",
        action="store_true",
        default=True,
        help="Enable new quality control system (default: True)",
    )
    parser.add_argument(
        "--max-quality-attempts",
        type=int,
        default=5,
        help="Maximum attempts to achieve acceptable quality (default: 5)",
    )
    parser.add_argument(
        "--show-parameter-reasoning", action="store_true", help="Display model's reasoning for parameter choices"
    )
