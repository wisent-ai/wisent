"""Launch the Wisent Gradio application."""

import gradio as gr
from wisent.app.ui.interface import build_interface
from wisent.core.utils.config_tools import constants as _C


_APP_TITLE = "Wisent - World's Best AI through Representation Engineering"


_FONT_HEAD = (
    f'<link href="{_C.WISENT_FONT_CDN_URL}" rel="stylesheet" '
    f'onload="console.log(\'WISENT: font loaded from {_C.WISENT_FONT_CDN_URL}\')" '
    f'onerror="console.error(\'WISENT: font FAILED to load from {_C.WISENT_FONT_CDN_URL}\')">'
    '<script>'
    'document.fonts.ready.then(function(){'
    'var has=document.fonts.check("16px Hubot Sans");'
    'console.log("WISENT: document.fonts.ready — Hubot Sans available:",has);'
    'if(!has)console.warn("WISENT: Hubot Sans not available, browser will use default font")'
    '});'
    '</script>'
)


def _build_theme():
    """Build Wisent theme with mint accents, supporting light and dark modes."""
    return gr.themes.Base(
        font=[gr.themes.Font(_C.WISENT_FONT_PRIMARY)],
        primary_hue=gr.themes.Color(
            c50=_C.WISENT_COLOR_MINT_LIGHT,
            c100=_C.WISENT_COLOR_MINT_LIGHT,
            c200=_C.WISENT_COLOR_MINT,
            c300=_C.WISENT_COLOR_MINT,
            c400=_C.WISENT_COLOR_MINT,
            c500=_C.WISENT_COLOR_MINT_DARK,
            c600=_C.WISENT_COLOR_MINT_DARK,
            c700=_C.WISENT_COLOR_MINT_ACCENT_DARK,
            c800=_C.WISENT_COLOR_MINT_ACCENT_DARK,
            c900=_C.WISENT_COLOR_MINT_ACCENT_DARK,
            c950=_C.WISENT_COLOR_MINT_ACCENT_DARK,
        ),
        neutral_hue=gr.themes.Color(
            c50=_C.WISENT_COLOR_LIGHT_BG,
            c100=_C.WISENT_COLOR_TEXT_LIGHT,
            c200=_C.WISENT_COLOR_NEUTRAL_200,
            c300=_C.WISENT_COLOR_TEXT_MUTED,
            c400=_C.WISENT_COLOR_NEUTRAL_400,
            c500=_C.WISENT_COLOR_NEUTRAL_500,
            c600=_C.WISENT_COLOR_LIGHT_TEXT_MUTED,
            c700=_C.WISENT_COLOR_DARK_SURFACE,
            c800=_C.WISENT_COLOR_CHARCOAL,
            c900=_C.WISENT_COLOR_DARK_BG,
            c950=_C.WISENT_COLOR_LIGHT_TEXT,
        ),
    ).set(
        body_background_fill=_C.WISENT_COLOR_LIGHT_BG,
        body_background_fill_dark=_C.WISENT_COLOR_DARK_BG,
        body_text_color=_C.WISENT_COLOR_LIGHT_TEXT,
        body_text_color_dark=_C.WISENT_COLOR_TEXT_LIGHT,
        block_background_fill=_C.WISENT_COLOR_LIGHT_SURFACE,
        block_background_fill_dark=_C.WISENT_COLOR_CHARCOAL,
        block_label_text_color=_C.WISENT_COLOR_MINT_ACCENT_DARK,
        block_label_text_color_dark=_C.WISENT_COLOR_MINT,
        block_title_text_color=_C.WISENT_COLOR_MINT_ACCENT_DARK,
        block_title_text_color_dark=_C.WISENT_COLOR_MINT,
        button_primary_background_fill=_C.WISENT_COLOR_MINT_ACCENT_DARK,
        button_primary_background_fill_dark=_C.WISENT_COLOR_MINT,
        button_primary_text_color=_C.WISENT_COLOR_LIGHT_SURFACE,
        button_primary_text_color_dark=_C.WISENT_COLOR_CHARCOAL,
        input_background_fill=_C.WISENT_COLOR_LIGHT_SURFACE,
        input_background_fill_dark=_C.WISENT_COLOR_DARK_SURFACE,
        border_color_primary=_C.WISENT_COLOR_MINT_DARK,
        border_color_primary_dark=_C.WISENT_COLOR_MINT_DARK,
    )


_CARD_CSS = (
    f".preset-card {{ cursor:pointer; text-align:center; "
    f"display:flex; flex-direction:column; align-items:center; "
    f"gap:{_C.PRESET_CARD_GAP_PX}px; "
    f"padding:{_C.PRESET_CARD_PADDING_PX}px; "
    f"border-radius:{_C.PRESET_CARD_RADIUS_PX}px; "
    f"border:{_C.PRESET_CARD_BORDER_WIDTH_PX}px solid "
    f"{_C.PRESET_CARD_BORDER_COLOR}; "
    f"background:{_C.PRESET_CARD_BG_COLOR}; "
    f"transition:border-color 0.2s; }} "
    f".preset-card:hover {{ "
    f"border-color:{_C.PRESET_CARD_HOVER_BORDER_COLOR}; }} "
    f".preset-card .pc-icon {{ "
    f"width:{_C.PRESET_CARD_ICON_SIZE_PX}px; "
    f"height:{_C.PRESET_CARD_ICON_SIZE_PX}px; "
    f"color:{_C.PRESET_CARD_ICON_COLOR}; }} "
    f".preset-card .pc-title {{ "
    f"font-weight:{_C.PRESET_CARD_TITLE_FONT_WEIGHT}; "
    f"font-size:{_C.PRESET_CARD_TITLE_SIZE_PX}px; "
    f"color:{_C.PRESET_CARD_TITLE_COLOR}; }} "
    f".preset-card .pc-desc {{ "
    f"font-size:{_C.PRESET_CARD_DESC_SIZE_PX}px; "
    f"color:{_C.PRESET_CARD_DESC_COLOR}; }} "
    f".dark .preset-card {{ "
    f"background:{_C.WISENT_COLOR_DARK_SURFACE}; "
    f"border-color:{_C.WISENT_COLOR_LIGHT_TEXT_MUTED}; }} "
    f".dark .preset-card:hover {{ "
    f"border-color:{_C.WISENT_COLOR_TEXT_MUTED}; }} "
    f".dark .preset-card .pc-icon {{ "
    f"color:{_C.WISENT_COLOR_TEXT_MUTED}; }} "
    f".dark .preset-card .pc-title {{ "
    f"color:{_C.WISENT_COLOR_TEXT_LIGHT}; }} "
    f".dark .preset-card .pc-desc {{ "
    f"color:{_C.WISENT_COLOR_NEUTRAL_400}; }} "
)

_APP_CSS = (
    ".output-box { font-family: monospace; white-space: pre-wrap; } "
    ".gradio-container { max-width: none !important; } "
    f"{_CARD_CSS}"
    f":root {{ "
    f"--input-text-color: {_C.WISENT_COLOR_LIGHT_TEXT}; "
    f"--checkbox-label-text-color: {_C.WISENT_COLOR_LIGHT_TEXT}; "
    f"--checkbox-label-text-color-selected: {_C.WISENT_COLOR_TEXT_LIGHT}; "
    f"--checkbox-label-background-fill: {_C.WISENT_COLOR_LIGHT_SURFACE}; "
    f"--checkbox-label-background-fill-hover: "
    f"{_C.WISENT_COLOR_MINT_LIGHT}; "
    f"--checkbox-label-background-fill-selected: "
    f"{_C.WISENT_COLOR_MINT_ACCENT_DARK}; "
    f"}} "
    f".dark {{ "
    f"--input-text-color: {_C.WISENT_COLOR_TEXT_LIGHT}; "
    f"--checkbox-label-text-color: {_C.WISENT_COLOR_TEXT_LIGHT}; "
    f"--checkbox-label-text-color-selected: {_C.WISENT_COLOR_CHARCOAL}; "
    f"--checkbox-label-background-fill: {_C.WISENT_COLOR_DARK_SURFACE}; "
    f"--checkbox-label-background-fill-hover: {_C.WISENT_COLOR_CHARCOAL}; "
    f"--checkbox-label-background-fill-selected: {_C.WISENT_COLOR_MINT}; "
    f"}} "
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
        "server_name": _C.GRADIO_SERVER_HOST,
        "server_port": _C.GRADIO_SERVER_PORT,
    }
    defaults.update(kwargs)
    app.launch(
        theme=_build_theme(), css=_APP_CSS, head=_FONT_HEAD,
        ssr_mode=False, **defaults,
    )


if __name__ == "__main__":
    launch()
