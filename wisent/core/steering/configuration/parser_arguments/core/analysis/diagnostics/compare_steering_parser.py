"""Parser setup for the 'compare-steering' command."""


def setup_compare_steering_parser(parser):
    """Set up the compare-steering command parser.

    Args:
        parser: argparse subparser for compare-steering.
    """
    parser.add_argument(
        "objects",
        nargs="+",
        type=str,
        help="Paths to .pt steering objects to compare (at least 2)",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer list to compare (default: all common layers)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for JSON results (default: print only)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="both",
        choices=["json", "table", "both"],
        help="Output format: json, table, or both (default: both)",
    )
