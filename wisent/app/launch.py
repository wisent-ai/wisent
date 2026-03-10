"""Launch the Wisent Gradio application."""

import gradio as gr
from wisent.app.ui.interface import build_interface
from wisent.core.utils.config_tools.constants import (
    GRADIO_SERVER_PORT,
    GRADIO_SERVER_HOST,
)


_APP_TITLE = "Wisent - AI Safety & Alignment Toolkit"
_APP_CSS = ".output-box { font-family: monospace; white-space: pre-wrap; }"


def create_app() -> gr.Blocks:
    """Create and return the Gradio Blocks application."""
    with gr.Blocks(title=_APP_TITLE) as app:
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
    app.launch(
        theme=gr.themes.Soft(),
        css=_APP_CSS,
        **defaults,
    )


if __name__ == "__main__":
    launch()
