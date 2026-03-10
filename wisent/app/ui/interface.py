"""Build the complete tabbed Gradio interface for Wisent.

Creates a model selector header and one tab per command group, each
containing sub-tabs for individual commands.
"""

import os
import gradio as gr
from wisent.core.utils.config_tools.constants import (
    GRADIO_MODEL_PLACEHOLDER,
    WISENT_LOGO_FILENAME,
    WISENT_LOGO_DISPLAY_WIDTH,
)
from wisent.app.core.groups import get_command_groups
from wisent.app.ui.command_tab import build_command_tab, build_subparser_tab
from wisent.app.ui.wizard import build_wizard_tab

_SUBPARSER_COMMANDS = frozenset({"optimize-steering"})


def _find_logo():
    """Locate the logo file relative to the app package."""
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(app_dir, WISENT_LOGO_FILENAME)
    if os.path.exists(path):
        return path
    return None


def build_interface():
    """Construct the full Gradio interface inside an active gr.Blocks context."""
    logo_path = _find_logo()
    if logo_path:
        gr.Image(
            value=logo_path,
            show_label=False,
            interactive=False,
            width=WISENT_LOGO_DISPLAY_WIDTH,
            container=False,
        )
    gr.Markdown(
        "# Wisent\n"
        "### AI Safety & Alignment Toolkit\n"
        "Select a category tab, choose a command, fill in parameters, "
        "and click Run."
    )

    with gr.Row():
        gr.Textbox(
            label="Model Name (shared across commands)",
            value="",
            placeholder=GRADIO_MODEL_PLACEHOLDER,
            interactive=True,
            elem_id="global-model",
        )

    groups = get_command_groups()

    with gr.Tabs():
        with gr.Tab(label="Wizard"):
            build_wizard_tab()
        for group in groups:
            with gr.Tab(label=group.label):
                gr.Markdown(f"*{group.description}*")
                with gr.Tabs():
                    for cmd in group.commands:
                        with gr.Tab(label=cmd.name):
                            if cmd.name in _SUBPARSER_COMMANDS:
                                build_subparser_tab(cmd)
                            else:
                                build_command_tab(cmd)
