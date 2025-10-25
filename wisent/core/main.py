"""
Main entry point for the Wisent CLI.

This module connects the argparse parser (wisent/core/parser_arguments/) to execution logic
and provides the main() function that serves as the CLI entry point.
"""

import sys
from wisent.core.parser_arguments import setup_parser
from wisent.core.branding import print_banner
from wisent.core.cli import execute_tasks


def main():
    """Main entry point for the Wisent CLI."""
    # Show banner
    print_banner("Wisent CLI", width=64, use_color=True)

    # Parse arguments
    parser = setup_parser()
    args = parser.parse_args()

    # If no command specified, show help
    if not hasattr(args, 'command') or args.command is None:
        parser.print_help()
        sys.exit(0)

    # Execute based on command
    if args.command == 'tasks':
        execute_tasks(args)
    else:
        print(f"\n✗ Command '{args.command}' is not yet implemented")
        sys.exit(1)


if __name__ == '__main__':
    main()
