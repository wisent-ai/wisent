"""
Parser arguments package for Wisent CLI.

This package contains argument parser definitions for each CLI command.
Each command has its own parser file for better organization and maintainability.
"""

from wisent.core.parser_arguments.main_parser import setup_parser

__all__ = ["setup_parser"]
