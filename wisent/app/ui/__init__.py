"""UI components for the Wisent Gradio app: interface, tabs, forms."""

from wisent.app.ui.interface import build_interface
from wisent.app.ui.command_tab import build_command_tab, build_subparser_tab

__all__ = ["build_interface", "build_command_tab", "build_subparser_tab"]
