"""Interactive command wizard for the Wisent Gradio interface.

Guides users through a two-level decision tree to find the right
CLI command for their use case.
"""

import gradio as gr
from wisent.app.ui.wiring.recommendations import RECOMMENDATIONS

_GOALS = [
    "Generate contrastive data",
    "Create steering vectors",
    "Steer a model at inference",
    "Evaluate model outputs",
    "Optimize parameters",
    "Analyze geometry and diagnostics",
    "Modify model weights",
    "Configure settings",
]

_SUBGOALS = {
    "Generate contrastive data": [
        "Generate synthetic contrastive pairs from a custom trait",
        "Generate pairs from an lm-eval benchmark task",
        "Generate model responses to evaluation questions",
        "Run the full synthetic pipeline end-to-end",
    ],
    "Create steering vectors": [
        "From an lm-eval task (full pipeline)",
        "From synthetic contrastive pairs",
        "From existing enriched pairs",
        "Discover the best steering direction automatically",
    ],
    "Steer a model at inference": [
        "Combine multiple steering vectors at inference",
        "Visualize how steering affects activation space",
        "Verify steered activations are aligned correctly",
        "Compare steering objects across traits",
    ],
    "Evaluate model outputs": [
        "Evaluate response quality with embedded evaluator",
        "Evaluate model refusal rate on harmful prompts",
    ],
    "Optimize parameters": [
        "Run all optimizations at once",
        "Optimize classification thresholds",
        "Optimize steering parameters (method, layer, strength)",
        "Find optimal training sample size",
        "Optimize weight modification parameters",
        "Manage cached optimization results",
        "Find the best steering method for a benchmark",
    ],
    "Analyze geometry and diagnostics": [
        "Diagnose contrastive pair quality",
        "Diagnose steering vector quality",
        "Check if a representation is linear",
        "Cluster benchmarks by direction similarity",
        "Search for unified goodness direction",
        "Run full Zwiad geometry analysis",
    ],
    "Modify model weights": [
        "Permanently modify model weights with steering",
        "Collect activations from contrastive pairs",
        "Train a unified goodness vector from benchmarks",
    ],
    "Configure settings": [
        "View and update inference settings",
        "Run evaluation tasks",
    ],
}

_RECOMMENDATIONS = RECOMMENDATIONS


def build_wizard_tab():
    """Build the interactive command wizard tab.

    Returns:
        Tuple of (go_button, cmd_state) for tab navigation wiring.
    """
    gr.Markdown(
        "## Command Wizard\n"
        "Answer two questions to find the right command for your use case."
    )
    goal = gr.Radio(label="What is your goal?", choices=_GOALS, value=None)
    subgoal = gr.Radio(
        label="More specifically?", choices=[], value=None, visible=False,
    )
    recommendation = gr.Markdown(
        value="*Select a goal above to get started.*", label="Recommendation",
    )
    cmd_state = gr.State(value=None)
    go_btn = gr.Button("Go to command", variant="primary", visible=False)

    def on_goal_change(selected_goal):
        if selected_goal and selected_goal in _SUBGOALS:
            choices = _SUBGOALS[selected_goal]
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

    def on_subgoal_change(selected_subgoal):
        if selected_subgoal and selected_subgoal in _RECOMMENDATIONS:
            cmd_name, description = _RECOMMENDATIONS[selected_subgoal]
            text = f"### Recommended: `{cmd_name}`\n\n{description}"
            return text, gr.update(visible=True), cmd_name
        return (
            "*Select a specific goal to see the recommendation.*",
            gr.update(visible=False),
            None,
        )

    goal.change(
        fn=on_goal_change, inputs=[goal],
        outputs=[subgoal, recommendation, go_btn, cmd_state],
    )
    subgoal.change(
        fn=on_subgoal_change, inputs=[subgoal],
        outputs=[recommendation, go_btn, cmd_state],
    )
    return go_btn, cmd_state
