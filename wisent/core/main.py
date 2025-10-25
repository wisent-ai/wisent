"""
Main entry point for the Wisent CLI.

This module connects the argparse parser (wisent/core/parser_arguments/) to command handlers
and provides the main() function that serves as the CLI entry point.
"""

import sys
from wisent.core.parser_arguments import setup_parser
from wisent.core.branding import print_banner


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

    # TODO: Implement command handlers
    # For now, just show that arguments were parsed successfully
    print(f"\n✓ Command parsed successfully: {args.command}")
    print(f"✓ Arguments: {vars(args)}")
    print(f"\nNote: Command implementation is pending. The CLI parser is working correctly.")


if __name__ == '__main__':
    main()
