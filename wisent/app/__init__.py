"""Wisent Gradio web interface.

Provides a browser-based UI for all Wisent CLI commands using dynamic
form generation from argparse parsers.
"""

from wisent.app.launch import launch

__all__ = ["launch"]
