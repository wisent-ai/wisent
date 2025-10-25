"""Parser setup for the 'monitor' command."""


def setup_monitor_parser(parser):
    """Set up the monitor subcommand parser."""
    parser.add_argument("--memory-info", action="store_true", help="Show current memory usage information")
    parser.add_argument("--system-info", action="store_true", help="Show system information and capabilities")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--test-gpu", action="store_true", help="Test GPU availability and memory")
    parser.add_argument("--continuous", action="store_true", help="Continuous monitoring mode (Ctrl+C to stop)")
    parser.add_argument("--interval", type=float, default=1.0, help="Monitoring interval in seconds (default: 1.0)")
    parser.add_argument("--export-csv", type=str, default=None, help="Export monitoring data to CSV file")
    parser.add_argument(
        "--duration", type=int, default=60, help="Duration for continuous monitoring in seconds (default: 60)"
    )
    parser.add_argument("--track-gpu", action="store_true", help="Include GPU monitoring (requires CUDA)")
    parser.add_argument("--detailed", action="store_true", help="Show detailed monitoring information")
