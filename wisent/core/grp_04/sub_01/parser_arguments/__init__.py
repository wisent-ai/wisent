import os as _os
_base = _os.path.dirname(__file__)
for _root, _dirs, _files in _os.walk(_base):
    _dirs[:] = sorted(d for d in _dirs if d.startswith(("grp_", "sub_", "mid_")))
    if _root != _base:
        __path__.append(_root)

"""
Parser arguments package for Wisent CLI.

This package contains argument parser definitions for each CLI command.
Each command has its own parser file for better organization and maintainability.
"""

from wisent.core.parser_arguments.main_parser import setup_parser

__all__ = ["setup_parser"]
