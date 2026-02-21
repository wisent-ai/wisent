"""
Parser for modify-weights command.

This command modifies model weights permanently using steering vectors.
"""

import argparse



from wisent.core.parser_arguments.other.modify_weights_parser_basic import setup_basic_modify_args
from wisent.core.parser_arguments.other.modify_weights_parser_advanced import setup_advanced_modify_args


def setup_modify_weights_parser(parser: argparse.ArgumentParser) -> None:
    """Set up the modify-weights command parser."""
    setup_basic_modify_args(parser)
    setup_advanced_modify_args(parser)
