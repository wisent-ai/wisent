"""Interactive command wizard for the Wisent Gradio interface.

Shows natural-language use-case presets that navigate directly to
the right command tab.
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
        "Pick a use case below, or browse the tabs to find a command."
    )

    preset_labels = [label for label, _cmd, _desc in PRESETS]
    preset_map = {label: (cmd, desc) for label, cmd, desc in PRESETS}

    preset_radio = gr.Radio(
        label="Use cases",
        choices=preset_labels,
        value=None,
    )
    recommendation = gr.Markdown(value="", label="Details")
    cmd_state = gr.State(value=None)
    go_btn = gr.Button("Go to command", variant="primary", visible=False)

    def on_preset_change(selected):
        if selected and selected in preset_map:
            cmd_name, description = preset_map[selected]
            text = f"### `{cmd_name}`\n\n{description}"
            return text, gr.update(visible=True), cmd_name
        return "", gr.update(visible=False), None

    preset_radio.change(
        fn=on_preset_change, inputs=[preset_radio],
        outputs=[recommendation, go_btn, cmd_state],
    )
    return go_btn, cmd_state
