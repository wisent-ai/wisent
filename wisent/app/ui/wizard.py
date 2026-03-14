"""Interactive command wizard for the Wisent Gradio interface.

Shows preset use-case cards at the top, followed by a multi-step
goal/subgoal wizard for browsing all commands.
"""

import os

import gradio as gr
from wisent.core.utils.config_tools.constants import INDEX_FIRST
from wisent.app.ui.wiring.recommendations import (
    GOALS, SUBGOALS, RECOMMENDATIONS, PRESETS,
)

_ICONS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "icons",
)


def _load_icon_svg(filename):
    """Load an SVG icon file and return its markup."""
    path = os.path.join(_ICONS_DIR, filename)
    with open(path) as fh:
        return fh.read().strip()


def _card_html(icon_svg, title, description):
    """Build the HTML for a single preset card."""
    return (
        f'<div class="preset-card">'
        f'<div class="pc-icon">{icon_svg}</div>'
        f'<div class="pc-title">{title}</div>'
        f'<div class="pc-desc">{description}</div>'
        f'</div>'
    )


def build_wizard_tab():
    """Build the wizard tab with preset cards and step-by-step flow.

    Returns:
        Tuple of (go_button, cmd_state) for tab navigation wiring.
    """
    cmd_state = gr.State(value=None)
    recommendation = gr.Markdown(value="")
    go_btn = gr.Button("Go to command", variant="primary", visible=False)

    gr.Markdown("### Quick start")
    with gr.Row(equal_height=True):
        for icon_file, label, cmd, desc in PRESETS:
            icon_svg = _load_icon_svg(icon_file)
            with gr.Column(min_width=INDEX_FIRST):
                card = gr.HTML(value=_card_html(icon_svg, label, desc))
                card.click(
                    fn=_make_preset_handler(cmd, label),
                    inputs=[],
                    outputs=[recommendation, go_btn, cmd_state],
                )

    gr.Markdown("---\n### Or browse by category")
    goal = gr.Radio(label="What is your goal?", choices=GOALS, value=None)
    subgoal = gr.Radio(
        label="More specifically?", choices=[], value=None, visible=False,
    )

    goal.change(
        fn=_on_goal_change, inputs=[goal],
        outputs=[subgoal, recommendation, go_btn, cmd_state],
    )
    subgoal.change(
        fn=_on_subgoal_change, inputs=[subgoal],
        outputs=[recommendation, go_btn, cmd_state],
    )
    return go_btn, cmd_state


def _make_preset_handler(cmd_name, title):
    """Create a click handler for a preset card."""
    def handler():
        text = f"### `{cmd_name}`\n\n{title}"
        return text, gr.update(visible=True), cmd_name
    return handler


def _on_goal_change(selected_goal):
    """Update subgoal choices when a goal is selected."""
    if selected_goal and selected_goal in SUBGOALS:
        choices = SUBGOALS[selected_goal]
        return (
            gr.update(choices=choices, value=None, visible=True),
            "*Now select a more specific goal.*",
            gr.update(visible=False),
            None,
        )
    return (
        gr.update(choices=[], visible=False),
        "",
        gr.update(visible=False),
        None,
    )


def _on_subgoal_change(selected_subgoal):
    """Show recommendation when a subgoal is selected."""
    if selected_subgoal and selected_subgoal in RECOMMENDATIONS:
        cmd_name, description = RECOMMENDATIONS[selected_subgoal]
        text = f"### `{cmd_name}`\n\n{description}"
        return text, gr.update(visible=True), cmd_name
    return (
        "*Select a specific goal to see the recommendation.*",
        gr.update(visible=False),
        None,
    )
