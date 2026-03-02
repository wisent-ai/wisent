"""Parser setup for the 'optimization-cache' command."""


def setup_optimization_cache_parser(parser):
    """Set up the optimization-cache subcommand parser."""
    # Create subparsers for different cache actions
    cache_subparsers = parser.add_subparsers(dest="cache_action", help="Cache management actions")

    # List cached results
    list_parser = cache_subparsers.add_parser(
        "list", help="List cached optimization results"
    )
    list_parser.add_argument("--model", type=str, default=None, help="Filter by model name")
    list_parser.add_argument("--task", type=str, default=None, help="Filter by task name")

    # Show details of a cached result
    show_parser = cache_subparsers.add_parser(
        "show", help="Show details of a cached optimization result"
    )
    show_parser.add_argument("model", type=str, help="Model name")
    show_parser.add_argument("task", type=str, help="Task name")
    show_parser.add_argument("--method", type=str, default="CAA", help="Steering method (default: CAA)")

    # Delete a cached result
    delete_parser = cache_subparsers.add_parser(
        "delete", help="Delete a cached optimization result"
    )
    delete_parser.add_argument("model", type=str, help="Model name")
    delete_parser.add_argument("task", type=str, help="Task name")
    delete_parser.add_argument("--method", type=str, default="CAA", help="Steering method (default: CAA)")

    # Clear all cached results
    clear_parser = cache_subparsers.add_parser(
        "clear", help="Clear all cached optimization results"
    )
    clear_parser.add_argument("--confirm", action="store_true", help="Confirm clearing all cached results")

    # Set default for a model/task
    set_default_parser = cache_subparsers.add_parser(
        "set-default", help="Set a cached result as the default for a model/task"
    )
    set_default_parser.add_argument("model", type=str, help="Model name")
    set_default_parser.add_argument("task", type=str, help="Task name")
    set_default_parser.add_argument("--method", type=str, default="CAA", help="Steering method to set as default (default: CAA)")

    # Get default for a model/task
    get_default_parser = cache_subparsers.add_parser(
        "get-default", help="Get the default optimization result for a model/task"
    )
    get_default_parser.add_argument("model", type=str, help="Model name")
    get_default_parser.add_argument("task", type=str, help="Task name")

    # Export cache to file
    export_parser = cache_subparsers.add_parser(
        "export", help="Export cache to a JSON file"
    )
    export_parser.add_argument("output", type=str, help="Output file path")

    # Import cache from file
    import_parser = cache_subparsers.add_parser(
        "import", help="Import cache from a JSON file"
    )
    import_parser.add_argument("input", type=str, help="Input file path")
    import_parser.add_argument("--merge", action="store_true", help="Merge with existing cache instead of replacing")
