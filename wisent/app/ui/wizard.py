"""Interactive command wizard for the Wisent Gradio interface.

Shows natural-language use-case presets as clickable buttons that
navigate directly to the right command tab.
"""

import gradio as gr
from wisent.app.ui.wiring.recommendations import PRESETS


def build_wizard_tab():
    """Build the interactive command wizard tab.

    Returns:
        Tuple of (go_button, cmd_state) for tab navigation wiring.
    """
    gr.Markdown(
        "## What do you want to do?\n"
        "Pick a use case to get started, or browse the tabs above."
    )

    cmd_state = gr.State(value=None)
    go_btn = gr.Button("Go to command", variant="primary", visible=False)
    recommendation = gr.Markdown(value="")

    preset_map = {label: (cmd, desc) for label, cmd, desc in PRESETS}

    for label, cmd, desc in PRESETS:
        btn = gr.Button(label, variant="secondary")
        btn.click(
            fn=_make_preset_handler(cmd, desc),
            inputs=[],
            outputs=[recommendation, go_btn, cmd_state],
        )

    return go_btn, cmd_state


def _make_preset_handler(cmd_name, description):
    """Create a click handler for a preset button."""
    def handler():
        text = f"### `{cmd_name}`\n\n{description}"
        return text, gr.update(visible=True), cmd_name
    return handler
