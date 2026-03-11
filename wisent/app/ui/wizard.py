"""Interactive command wizard for the Wisent Gradio interface.

Guides users through a two-level decision tree to find the right
CLI command for their use case.
"""

import gradio as gr

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

_RECOMMENDATIONS = {
    "Generate synthetic contrastive pairs from a custom trait": (
        "generate-pairs",
        "Generates contrastive pairs where the model answers the same "
        "prompts with and without a specified trait. Requires a model "
        "name and trait description. Output pairs train steering vectors.",
    ),
    "Generate pairs from an lm-eval benchmark task": (
        "generate-pairs-from-task",
        "Extracts contrastive pairs from an lm-eval harness task. The "
        "correct answer becomes the positive example and an incorrect "
        "answer becomes the negative.",
    ),
    "Generate model responses to evaluation questions": (
        "generate-responses",
        "Generates model responses to evaluation questions, optionally "
        "with steering applied. Useful for comparing steered vs "
        "unsteered outputs or collecting responses for later evaluation.",
    ),
    "Run the full synthetic pipeline end-to-end": (
        "synthetic",
        "Runs the complete synthetic pipeline: generates prompts, "
        "collects positive and negative responses, and saves enriched "
        "pairs ready for activation extraction.",
    ),
    "From an lm-eval task (full pipeline)": (
        "generate-vector-from-task",
        "Full pipeline: extracts pairs from an lm-eval task, collects "
        "activations, and trains a steering vector. One command from "
        "benchmark name to ready-to-use steering object.",
    ),
    "From synthetic contrastive pairs": (
        "generate-vector-from-synthetic",
        "Full pipeline from synthetic data: generates contrastive pairs "
        "for a trait, collects activations, and trains a steering vector.",
    ),
    "From existing enriched pairs": (
        "create-steering-vector",
        "Creates a steering vector from already-enriched contrastive "
        "pairs (with activations). Use when you have previously collected "
        "activations and want to retrain the vector.",
    ),
    "Discover the best steering direction automatically": (
        "discover-steering",
        "Searches for the optimal steering direction by trying multiple "
        "methods, layers, and hyperparameters. Returns the best "
        "steering configuration.",
    ),
    "Combine multiple steering vectors at inference": (
        "multi-steer",
        "Applies multiple steering vectors simultaneously during "
        "generation. Specify several vector paths with individual "
        "strengths to combine traits like truthfulness and helpfulness.",
    ),
    "Visualize how steering affects activation space": (
        "steering-viz",
        "Creates visualizations showing how steering affects the model "
        "activation space. Produces PCA/UMAP plots comparing steered "
        "vs unsteered activations and per-concept breakdowns.",
    ),
    "Verify steered activations are aligned correctly": (
        "verify-steering",
        "Verifies that a steered model produces activations aligned with "
        "the intended direction. Useful as a sanity check after creating "
        "or optimizing a steering vector.",
    ),
    "Compare steering objects across traits": (
        "compare-steering",
        "Compares multiple steering objects to analyze how different "
        "traits relate in activation space. Shows cosine similarity, "
        "overlap, and interference between directions.",
    ),
    "Evaluate response quality with embedded evaluator": (
        "evaluate-responses",
        "Evaluates generated responses using NLI, embedding similarity, "
        "or other built-in evaluators. Scores how well steered responses "
        "match desired behavior.",
    ),
    "Evaluate model refusal rate on harmful prompts": (
        "evaluate-refusal",
        "Measures how often a model refuses to answer harmful prompts. "
        "Useful for safety testing steered models to ensure refusal "
        "behavior is preserved.",
    ),
    "Run all optimizations at once": (
        "optimize",
        "Runs the complete optimization suite: classification thresholds, "
        "steering parameters, and evaluation metrics. Best for optimizing "
        "everything without manual tuning.",
    ),
    "Optimize classification thresholds": (
        "optimize-classification",
        "Tunes MLP classifier hyperparameters for separating positive "
        "and negative activations: batch size, learning rate, hidden "
        "dimensions, and dropout.",
    ),
    "Optimize steering parameters (method, layer, strength)": (
        "optimize-steering",
        "Searches for the best steering method (CAA, TECZA, OSTRZE, etc.), "
        "target layer, and strength via Bayesian optimization.",
    ),
    "Find optimal training sample size": (
        "optimize-sample-size",
        "Determines the minimum number of contrastive pairs needed. "
        "Runs the pipeline with increasing sample sizes and finds the "
        "point of diminishing returns.",
    ),
    "Optimize weight modification parameters": (
        "optimize-weights",
        "Optimizes parameters for permanent weight modification: target "
        "layers, modification strength, and approach.",
    ),
    "Manage cached optimization results": (
        "optimization-cache",
        "View, clear, or export cached optimization results. Inspect "
        "previous runs or reset the cache before a fresh sweep.",
    ),
    "Find the best steering method for a benchmark": (
        "find-best-method",
        "Exhaustively trials every registered steering method on a "
        "benchmark and ranks them. Generates all contrastive pairs once, "
        "splits train/test, optimizes each method independently, and "
        "reports the winner with response diff and activation analysis.",
    ),
    "Diagnose contrastive pair quality": (
        "diagnose-pairs",
        "Analyzes contrastive pairs for quality issues: semantic "
        "divergence, length imbalance, duplication, and consistency. "
        "Run before training to catch data problems early.",
    ),
    "Diagnose steering vector quality": (
        "diagnose-vectors",
        "Analyzes a trained steering vector: activation separability, "
        "direction stability across layers, and interference with other "
        "directions in the model.",
    ),
    "Check if a representation is linear": (
        "check-linearity",
        "Tests whether a concept is represented linearly in model "
        "activations. High linearity means simple steering methods "
        "like CAA will work well for that trait.",
    ),
    "Cluster benchmarks by direction similarity": (
        "cluster-benchmarks",
        "Groups benchmarks whose steering directions are similar, "
        "revealing which tasks share geometry and enabling transfer "
        "of vectors between related tasks.",
    ),
    "Search for unified goodness direction": (
        "geometry-search",
        "Searches for a single steering direction that improves "
        "performance across multiple benchmarks simultaneously.",
    ),
    "Run full Zwiad geometry analysis": (
        "zwiad",
        "Runs comprehensive RepScan geometry analysis: layer sensitivity "
        "profiling, method comparison, concept-level breakdowns, and "
        "similarity heatmaps across the full model.",
    ),
    "Permanently modify model weights with steering": (
        "modify-weights",
        "Permanently applies steering to model weights so the effect "
        "persists without runtime steering. Saves a new model with "
        "the trait baked in.",
    ),
    "Collect activations from contrastive pairs": (
        "get-activations",
        "Extracts hidden-state activations from a model for each "
        "contrastive pair. Required before training steering vectors "
        "when building the pipeline step by step.",
    ),
    "Train a unified goodness vector from benchmarks": (
        "train-unified-goodness",
        "Trains a single steering vector from multiple benchmark tasks "
        "capturing a general goodness direction. Uses geometry search "
        "results to combine task-specific vectors.",
    ),
    "View and update inference settings": (
        "inference-config",
        "View or modify inference configuration: temperature, top-p, "
        "max tokens, and other generation parameters. Persisted and "
        "used by all generation commands.",
    ),
    "Run evaluation tasks": (
        "tasks",
        "Lists or runs evaluation tasks from the lm-eval harness. Check "
        "available benchmarks or execute a specific evaluation suite.",
    ),
}


def build_wizard_tab():
    """Build the interactive command wizard tab."""
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

    def on_goal_change(selected_goal):
        if selected_goal and selected_goal in _SUBGOALS:
            choices = _SUBGOALS[selected_goal]
            return (
                gr.update(choices=choices, value=None, visible=True),
                "*Now select a more specific goal.*",
            )
        return gr.update(choices=[], visible=False), ""

    def on_subgoal_change(selected_subgoal):
        if selected_subgoal and selected_subgoal in _RECOMMENDATIONS:
            cmd_name, description = _RECOMMENDATIONS[selected_subgoal]
            return (
                f"### Recommended command: `{cmd_name}`\n\n"
                f"{description}\n\n"
                f"Find this command in the tabs above, or run via CLI:\n"
                f"```\nwisent {cmd_name} --help\n```"
            )
        return "*Select a specific goal to see the recommendation.*"

    goal.change(fn=on_goal_change, inputs=[goal], outputs=[subgoal, recommendation])
    subgoal.change(fn=on_subgoal_change, inputs=[subgoal], outputs=[recommendation])
