"""Launch the Wisent Gradio application."""

import gradio as gr
from wisent.app.ui.interface import build_interface
from wisent.core.utils.config_tools.constants import (
    GRADIO_SERVER_PORT,
    GRADIO_SERVER_HOST,
    WISENT_COLOR_MINT,
    WISENT_COLOR_MINT_LIGHT,
    WISENT_COLOR_MINT_DARK,
    WISENT_COLOR_CHARCOAL,
    WISENT_COLOR_DARK_BG,
    WISENT_COLOR_DARK_SURFACE,
    WISENT_COLOR_TEXT_LIGHT,
    WISENT_COLOR_TEXT_MUTED,
)


_APP_TITLE = "Wisent - World's Best AI through Representation Engineering"


def _build_theme():
    """Build the custom Wisent dark theme with mint accents."""
    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50=WISENT_COLOR_MINT_LIGHT,
            c100=WISENT_COLOR_MINT_LIGHT,
            c200=WISENT_COLOR_MINT,
            c300=WISENT_COLOR_MINT,
            c400=WISENT_COLOR_MINT,
            c500=WISENT_COLOR_MINT,
            c600=WISENT_COLOR_MINT_DARK,
            c700=WISENT_COLOR_MINT_DARK,
            c800=WISENT_COLOR_MINT_DARK,
            c900=WISENT_COLOR_MINT_DARK,
            c950=WISENT_COLOR_MINT_DARK,
        ),
        neutral_hue=gr.themes.Color(
            c50=WISENT_COLOR_TEXT_LIGHT,
            c100=WISENT_COLOR_TEXT_MUTED,
            c200=WISENT_COLOR_TEXT_MUTED,
            c300=WISENT_COLOR_TEXT_MUTED,
            c400=WISENT_COLOR_DARK_SURFACE,
            c500=WISENT_COLOR_DARK_SURFACE,
            c600=WISENT_COLOR_CHARCOAL,
            c700=WISENT_COLOR_CHARCOAL,
            c800=WISENT_COLOR_DARK_BG,
            c900=WISENT_COLOR_DARK_BG,
            c950=WISENT_COLOR_DARK_BG,
        ),
    ).set(
        body_background_fill=WISENT_COLOR_DARK_BG,
        body_background_fill_dark=WISENT_COLOR_DARK_BG,
        body_text_color=WISENT_COLOR_TEXT_LIGHT,
        body_text_color_dark=WISENT_COLOR_TEXT_LIGHT,
        block_background_fill=WISENT_COLOR_CHARCOAL,
        block_background_fill_dark=WISENT_COLOR_CHARCOAL,
        block_label_text_color=WISENT_COLOR_MINT,
        block_label_text_color_dark=WISENT_COLOR_MINT,
        block_title_text_color=WISENT_COLOR_MINT,
        block_title_text_color_dark=WISENT_COLOR_MINT,
        button_primary_background_fill=WISENT_COLOR_MINT,
        button_primary_background_fill_dark=WISENT_COLOR_MINT,
        button_primary_text_color=WISENT_COLOR_CHARCOAL,
        button_primary_text_color_dark=WISENT_COLOR_CHARCOAL,
        input_background_fill=WISENT_COLOR_DARK_SURFACE,
        input_background_fill_dark=WISENT_COLOR_DARK_SURFACE,
        border_color_primary=WISENT_COLOR_MINT_DARK,
        border_color_primary_dark=WISENT_COLOR_MINT_DARK,
    )


_APP_CSS = (
    ".output-box { font-family: monospace; white-space: pre-wrap; } "
    ".gradio-container { max-width: none !important; }"
)


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
    app.launch(theme=_build_theme(), css=_APP_CSS, **defaults)


if __name__ == "__main__":
    launch()
