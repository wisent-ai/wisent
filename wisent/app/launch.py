"""Launch the Wisent Gradio application."""

import gradio as gr
from wisent.app.ui.interface import build_interface
from wisent.core.utils.config_tools.constants import (
    GRADIO_SERVER_PORT,
    GRADIO_SERVER_HOST,
)


def create_app() -> gr.Blocks:
    """Create and return the Gradio Blocks application."""
    with gr.Blocks(
        title="Wisent - AI Safety & Alignment Toolkit",
        theme=gr.themes.Soft(),
        css=".output-box { font-family: monospace; white-space: pre-wrap; }",
    ) as app:
        build_interface()
    return app


def launch(**kwargs):
    """Launch the Wisent Gradio application.

    Args:
        **kwargs: Forwarded to gr.Blocks.launch() (e.g. share, server_port).
    """
    app = create_app()
    defaults = {
        "server_name": GRADIO_SERVER_HOST,
        "server_port": GRADIO_SERVER_PORT,
    }
    defaults.update(kwargs)
    app.launch(**defaults)


if __name__ == "__main__":
    launch()
