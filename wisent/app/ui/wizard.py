"""Interactive command wizard for the Wisent Gradio interface.

Shows preset use-case buttons at the top, followed by a multi-step
goal/subgoal wizard for browsing all commands.
"""

import gradio as gr
from wisent.app.ui.wiring.recommendations import (
    GOALS, SUBGOALS, RECOMMENDATIONS, PRESETS,
)


def build_wizard_tab():
    """Build the wizard tab with presets and step-by-step flow.

    Returns:
        Tuple of (go_button, cmd_state) for tab navigation wiring.
    """
    cmd_state = gr.State(value=None)
    recommendation = gr.Markdown(value="")
    go_btn = gr.Button("Go to command", variant="primary", visible=False)

    gr.Markdown("### Quick start")
    with gr.Row():
        for label, cmd, desc in PRESETS:
            btn = gr.Button(label, variant="secondary", size="sm")
            btn.click(
                fn=_make_preset_handler(cmd, desc),
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


def _make_preset_handler(cmd_name, description):
    """Create a click handler for a preset button."""
    def handler():
        text = f"### `{cmd_name}`\n\n{description}"
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
